#!/usr/bin/env python3
# cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
# fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
# a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
# !/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import npfl138

npfl138.require_version("2425.9")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=256, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=150, type=int, help="Number of epochs.")

parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=384, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=100, type=int, help="Show results every given batch.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0, help="Dropout rate.")

parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")


class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""

    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        # - `self._project_encoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs.
        # - `self._project_decoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs
        # - `self._output_layer` as a linear layer with `attention_dim` inputs and 1 output
        self._project_encoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._project_decoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._output_layer = torch.nn.Linear(attention_dim, 1)

    def setup_memory(self, encoded):
        self._encoded = encoded
        # Pass the `encoded` through the `self._project_encoder_layer` and store
        # the result as `self._encoded_projected`.
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        # Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        projected_decoder_state = self._project_decoder_layer(states[0] if isinstance(states, tuple) else states)
        # - Sum the two projections. However, you have to deal with the fact that the first projection has
        #   shape `[batch_size, input_sequence_len, attention_dim]`, while the second projection has
        #   shape `[batch_size, attention_dim]`. The best solution is capable of creating the sum
        #   directly without creating any intermediate tensor.
        attention_scores = self._output_layer(
            torch.tanh(self._encoded_projected + projected_decoder_state.unsqueeze(1))).squeeze(-1)
        # - Pass the sum through the `torch.tanh` and then through the `self._output_layer`.
        # - Then, run softmax activation, generating `weights`.
        weights = torch.nn.functional.softmax(attention_scores, dim=1).unsqueeze(1)
        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input word had.
        attention = torch.bmm(weights, self._encoded).squeeze(1)
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.
        concatenated_input = torch.cat([inputs, attention], dim=-1)
        return self._cell(concatenated_input, states)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        self._source_embedding = torch.nn.Embedding(len(self._source_vocab), args.cle_dim,
                                                    padding_idx=MorphoDataset.PAD)
        self._source_rnn = torch.nn.GRU(args.cle_dim, args.rnn_dim, bidirectional=True, batch_first=True)

        self._target_rnn_cell = WithAttention(torch.nn.GRUCell(2 * args.rnn_dim, args.rnn_dim), args.rnn_dim)
        self._target_output_layer = torch.nn.Linear(args.rnn_dim, len(self._target_vocab))

        if not args.tie_embeddings:
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab), args.rnn_dim,
                                                        padding_idx=MorphoDataset.PAD)
        else:
            assert args.cle_dim == args.rnn_dim, "When tying embeddings, cle_dim and rnn_dim must match."

            def tied_embedding(indices):
                return torch.nn.functional.embedding(indices, self._target_output_layer.weight) * (args.rnn_dim ** 0.5)

            self._target_embedding = tied_embedding

        # Add dropout
        self._dropout = torch.nn.Dropout(args.dropout)

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, words: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        # words= x = words.long()
        embedded = self._source_embedding(words)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, words.ne(MorphoDataset.PAD).sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self._source_rnn(packed_embedded)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = output[:, :, :self._source_rnn.hidden_size] + output[:, :, self._source_rnn.hidden_size:]
        return self._dropout(output)

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        decoder_inputs = torch.cat([torch.full_like(targets[:, :1], MorphoDataset.BOW), targets[:, :-1]], dim=1)
        self._target_rnn_cell.setup_memory(encoded)
        embedded_inputs = self._target_embedding(decoder_inputs)
        batch_size = encoded.shape[0]
        max_target_len = decoder_inputs.shape[1]
        hidden = encoded.new_zeros(batch_size, self._target_rnn_cell._cell.hidden_size)
        outputs = []
        for i in range(max_target_len):
            hidden = self._target_rnn_cell(embedded_inputs[:, i], hidden)
            hidden = self._dropout(hidden)
            outputs.append(hidden)
        outputs = torch.stack(outputs, dim=1)
        logits = self._target_output_layer(outputs)
        return logits.permute(0, 2, 1)

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]
        self._target_rnn_cell.setup_memory(encoded)
        index = 0
        inputs = torch.full((batch_size,), MorphoDataset.BOW, dtype=torch.int64, device=encoded.device)
        states = encoded.new_zeros(batch_size, self._target_rnn_cell._cell.hidden_size)
        results = []
        result_lengths = torch.full((batch_size,), max_length, dtype=torch.int64, device=encoded.device)

        while index < max_length and torch.any(result_lengths == max_length):
            embedded_inputs = self._target_embedding(inputs)
            hidden = self._target_rnn_cell(embedded_inputs, states)
            states = hidden
            output = self._target_output_layer(hidden)
            predictions = torch.argmax(output, dim=-1)
            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1
            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if self.training:
            y_pred = y_pred.argmax(dim=-2)
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred_padded = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        exact_match = torch.all((y_pred_padded == y) | (y == MorphoDataset.PAD), dim=-1)
        predictions = exact_match.long()
        targets = torch.ones_like(predictions)
        self.metrics["accuracy"].update(predictions, targets)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def train_step(self, xs, y):
        # Move inputs and targets to the device
        xs = [x.to(self.device) for x in xs]
        y = y.to(self.device)
        result = super().train_step(xs, y)

        self._batches += 1
        if self._batches % self._show_results_every_batch == 0:
            self.log_console(f"Batch {self._batches}, Loss: {result['loss']:.4f}")

        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.forward(*xs)
            # Trim the predictions at the first EOW
            batch = [lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0] for lemma in batch]
            return [lemma.cpu().numpy() for lemma in batch] if as_numpy else [lemma.cpu() for lemma in batch]


def perform_final_prediction(model, args, morpho, test_loader, epoch=None):
    """Generates and returns the final test set predictions as a list of strings."""
    all_predicted_lemmas = []

    # Ensure the model is in evaluation mode and on the correct device (GPU)
    model.eval()
    device = next(model.parameters()).device  # Get the device of the model

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                inputs = batch[0]
            else:
                raise ValueError("Batch from test_loader is not a non-empty tuple or list.")

            # Move inputs to the correct device (GPU)
            inputs = inputs.to(device)

            # Perform predictions on GPU
            predictions = model.predict_step((inputs,))
            for prediction in predictions:
                lemma = "".join(morpho.test.lemmas.char_vocab.strings(prediction))
                all_predicted_lemmas.append(lemma)

    # Save predictions to file
    filename = "lemmatizer_competition.txt" if epoch is None else f"lemmatizer_competition_epoch_{epoch}.txt"
    predictions_path = os.path.join(args.logdir, filename)
    os.makedirs(args.logdir, exist_ok=True)

    # Write predictions to file
    with open(predictions_path, "w", encoding="utf-8") as predictions_file:
        prediction_index = 0
        for gold_sentence in morpho.test.lemmas.strings:
            for _ in gold_sentence:
                if prediction_index < len(all_predicted_lemmas):
                    predictions_file.write(all_predicted_lemmas[prediction_index] + "\n")
                    prediction_index += 1
                else:
                    predictions_file.write("EOW\n")
            predictions_file.write("\n")

    print(f"Predictions saved to: {predictions_path}")
    return all_predicted_lemmas, predictions_path


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        # TODO: Return `example["words"]` as inputs and `example["lemmas"]` as targets.
        return example["words"], example["lemmas"]

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples generated by `transform`.
        words, lemmas = zip(*batch)
        word_indices = [self.dataset.words.char_vocab.indices(word) for sentence in words for word in sentence]
        words_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(w, dtype=torch.long) for w in word_indices], batch_first=True
        )

        # Flatten lemma sequences and map characters to indices, appending EOW
        lemma_indices = [
            self.dataset.lemmas.char_vocab.indices(lemma) + [MorphoDataset.EOW]
            for sentence in lemmas for lemma in sentence
        ]
        lemmas_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(l, dtype=torch.long) for l in lemma_indices], batch_first=True
        )

        # Return input and target tensors based on training flag
        inputs = (words_tensor, lemmas_tensor) if self._training else words_tensor
        return inputs, lemmas_tensor


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Add early stopper #
    class EarlyStopper:
        def __init__(self, patience=10):
            self.patience = patience
            self.counter = 0
            self.max_validation_accuracy = 0.0

        def early_stop(self, validation_accuracy):
            if validation_accuracy > (self.max_validation_accuracy + .0004):
                self.max_validation_accuracy = validation_accuracy
                print('So far, this was our best model based on validation accuracy!')
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    early_stopper = EarlyStopper(patience=9)

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Prepare the data for training and evaluation.
    train_dataset = TrainableDataset(morpho.train, training=True)
    dev_dataset = TrainableDataset(morpho.dev, training=False)
    test_dataset = TrainableDataset(morpho.test, training=False)

    train_loader = train_dataset.dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = dev_dataset.dataloader(batch_size=args.batch_size)
    test_loader = test_dataset.dataloader(batch_size=args.batch_size)

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model and move it to the selected device.
    model = Model(args, morpho.train).to(device)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD).to(device),
        metrics={
            "accuracy": torchmetrics.Accuracy(task='multiclass', num_classes=len(morpho.train.lemmas.char_vocab)).to(
                device)},
        logdir=args.logdir,
    )

    # Assuming you've set up your optimizer as part of the model configuration:
    optimizer = torch.optim.Adam(model.parameters())

    # Define the scheduler here
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    all_logs = []
    best_dev_accuracy = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.logdir, "best_model.pth")

    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch + 1} ---")

        logs = model.fit(train_loader, dev=dev_loader, epochs=1)
        all_logs.append(logs)

        train_loss = logs.get('train_loss', 0)
        train_accuracy = logs.get('train_accuracy', 0)
        dev_accuracy = logs.get('dev_accuracy', 0)
        dev_loss = logs.get('dev_loss', 0)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"--- Best model saved at epoch {best_epoch} with Dev Accuracy: {best_dev_accuracy:.4f} ---")

            if best_dev_accuracy > .97:
                print(f"\n--- Generating Predictions on Test Set (Epoch {epoch + 1}) ---")
                _, predictions_path = perform_final_prediction(model, args, morpho, test_loader, epoch=epoch + 1)
                print("--- End of Prediction ---\n")

        # checking for early stopping
        # if early_stopper.early_stop(dev_accuracy):
        # print('Early Stopping happened.')
        # break

        scheduler.step(dev_accuracy)
        print('...........................................................................')

    print(f"\n--- Generating Predictions on Test Set (Epoch {epoch + 1}) ---")
    _, predictions_path = perform_final_prediction(model, args, morpho, test_loader, epoch=epoch + 1)
    print("--- End of Prediction ---\n")

    return all_logs[-1] if all_logs else {}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
