#!/usr/bin/env python3
# cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
# fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
# a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
import argparse
import datetime
import os
import re

import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchmetrics
from torch.utils.data import Dataset

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--max_sentences", default=10, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=45, type=int, help="Random seed.")
parser.add_argument("--show_predictions", default=True, action="store_true", help="Show predicted tag sequences.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.

def check_bio_format(tags: list[str]) -> bool:
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        if tag.startswith("I-"):
            if i == 0:
                return False
            prev_tag = tags[i - 1]
            if prev_tag == "O":
                return False
            if prev_tag[2:] != tag[2:]:
                return False
        elif not tag.startswith("B-"):
            return False
    return True

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._show_predictions = args.show_predictions

        tag_vocab = list(train.tags.string_vocab)
        self._tag_vocab = tag_vocab
        num_tags = len(tag_vocab)

        A = torch.zeros((num_tags, num_tags), dtype=torch.bool)
        for i, t1 in enumerate(tag_vocab):
            for j, t2 in enumerate(tag_vocab):
                if t1 == "O":
                    A[i, j] = (t2 == "O") or t2.startswith("B-")
                elif t1.startswith("B-"):
                    if t2.startswith("I-"):
                        A[i, j] = t1[2:] == t2[2:]
                    else:
                        A[i, j] = (t2 == "O") or t2.startswith("B-")
                elif t1.startswith("I-"):
                    if t2.startswith("I-"):
                        A[i, j] = t1[2:] == t2[2:]
                    else:
                        A[i, j] = (t2 == "O") or t2.startswith("B-")
                elif t1 == "[PAD]" or t1 == "[UNK]":
                    A[i, j] = True # Allow any transition from PAD and UNK

        self.register_buffer("_A", A)
        #print("Corrected Transition Matrix A:") # Debugging
        #for i in range(num_tags):
            #print(f"{tag_vocab[i]}: {[tag_vocab[j] for j, allowed in enumerate(A[i]) if allowed]}")

        self._O_idx = tag_vocab.index("O")

        self._word_embedding = torch.nn.Embedding(
            num_embeddings=len(train.words.string_vocab),
            embedding_dim=args.we_dim
        )

        rnn_class = torch.nn.LSTM if args.rnn == "LSTM" else torch.nn.GRU
        self._word_rnn = rnn_class(
            input_size=args.we_dim,
            hidden_size=args.rnn_dim,
            bidirectional=True,
            batch_first=True
        )

        # Correct the input dimension of the output layer
        self._output_layer = torch.nn.Linear(args.rnn_dim, num_tags)

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        embedded = self._word_embedding(word_ids)
        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self._word_rnn(packed)
        hidden, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        # For bidirectional RNN, the output is [batch_size, seq_len, hidden_dim * 2]
        # We should sum the forward and backward parts
        rnn_dim = self._word_rnn.hidden_size
        hidden = hidden[:, :, :rnn_dim] + hidden[:, :, rnn_dim:]
        output = self._output_layer(hidden)
        output = output.permute(0, 2, 1)
        return output

    # ... (rest of your Model class) ...

    def constrained_decoding(self, logits: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        batch_size, num_tags, seq_len = logits.shape
        device = logits.device

        log_probs = torch.log_softmax(logits, dim=1).permute(0, 2, 1)  # [B, T, C]
        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1)  # [B]
        A_log = torch.where(self._A,
                            torch.zeros_like(self._A, dtype=torch.float32, device=device),
                            torch.full_like(self._A, -1e9, dtype=torch.float32, device=device))

        dp = torch.full((batch_size, seq_len, num_tags), -1e9, dtype=torch.float32, device=device)
        bp = torch.zeros((batch_size, seq_len, num_tags), dtype=torch.long, device=device)

        allowed_start = [j for j, tag in enumerate(self._tag_vocab) if tag == "O" or tag.startswith("B-")]

        dp[:, 0, :] = -1e9
        dp[:, 0, allowed_start] = log_probs[:, 0, allowed_start]

        for t in range(1, seq_len):
            prev_dp = dp[:, t - 1, :].unsqueeze(2)  # [B, num_tags, 1]
            trans_scores = A_log.unsqueeze(0)      # [1, num_tags, num_tags]
            scores = prev_dp + trans_scores          # [B, num_tags, num_tags]
            best_scores, best_indices = scores.max(dim=1)  # [B, num_tags]
            dp[:, t, :] = best_scores + log_probs[:, t, :]
            bp[:, t, :] = best_indices

        preds = torch.full((batch_size, seq_len), MorphoDataset.PAD, dtype=torch.long, device=device)
        for b in range(batch_size):
            L = lengths[b].item()  # valid length for sample b
            last_tag = dp[b, L - 1, :].argmax().item()
            preds[b, L - 1] = last_tag
            for t in range(L - 2, -1, -1):
                preds[b, t] = bp[b, t + 1, preds[b, t + 1]]
            preds[b, L:] = MorphoDataset.PAD

        # Post-Processing: Adjust tokens to ensure BIO correctness.
        for b in range(batch_size):
            L = lengths[b].item()
            if L > 0:
                first_tag = preds[b, 0].item()
                first_tag_str = self._tag_vocab[first_tag]
                if first_tag_str.startswith("I-"):
                    new_tag = "B-" + first_tag_str[2:]
                    if new_tag in self._tag_vocab:
                        preds[b, 0] = self._tag_vocab.index(new_tag)

            for t in range(1, L):
                prev_str = self._tag_vocab[preds[b, t - 1].item()]
                curr_str = self._tag_vocab[preds[b, t].item()]
                if prev_str == "O" and curr_str.startswith("I-"):
                    new_tag = "B-" + curr_str[2:]
                    if new_tag in self._tag_vocab:
                        preds[b, t] = self._tag_vocab.index(new_tag)
        return preds
    
    
    def compute_metrics(self, y_pred, y, word_ids):
        self.metrics["accuracy"].update(y_pred, y)
        if self.training:
            return {"accuracy": self.metrics["accuracy"].compute()}

        # Perform greedy decoding.
        predictions_greedy = y_pred.argmax(dim=1)
        predictions_greedy.masked_fill_(word_ids == MorphoDataset.PAD, MorphoDataset.PAD)
        self.metrics["f1_greedy"].update(predictions_greedy, y)

        # Perform constrained decoding.
        predictions = self.constrained_decoding(y_pred, word_ids)
        predictions.masked_fill_(word_ids == MorphoDataset.PAD, MorphoDataset.PAD)
        self.metrics["f1_constrained"].update(predictions, y)

        if self._show_predictions:
            for pred_tags in predictions:
                readable_tags = [self.metrics["f1_constrained"]._labels[tag.item()] if tag != MorphoDataset.PAD else "[PAD]" for tag in pred_tags]
                print(*readable_tags)

        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    
    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.constrained_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            batch = [example[example != MorphoDataset.PAD] for example in batch]
            return batch

class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        word_ids = torch.tensor([self.dataset.words.string_vocab.index(word) for word in example["words"]], dtype=torch.int64)
        tag_ids = torch.tensor([self.dataset.tags.string_vocab.index(tag) for tag in example["tags"]], dtype=torch.int64)
        return word_ids, tag_ids

    def collate(self, batch):
        word_ids, tag_ids = zip(*batch)
        # TODO(tagger_we): Combine `word_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        word_ids = rnn_utils.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        # TODO(tagger_we): Process `tag_ids` analogously to `word_ids`.
        tag_ids = rnn_utils.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        return word_ids, tag_ids

def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data.
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(tagger_we): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO: Use `torch.nn.CrossEntropyLoss` to instantiate the loss function.
        # Pass `ignore_index=morpho.PAD` to the constructor to ignore padding tags
        # during loss computation; also pass `label_smoothing=args.label_smoothing`.
        loss=torch.nn.CrossEntropyLoss(
            ignore_index=MorphoDataset.PAD,
            label_smoothing=args.label_smoothing
        ),
        metrics={
            # TODO(tagger_we): Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
            # the first argument, `num_classes` set to the number of unique tags, and
            # again `ignore_index=morpho.PAD` to ignore the padded tags.
            "accuracy": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=len(morpho.train.tags.string_vocab),
                ignore_index=MorphoDataset.PAD
            ),
            # TODO: Create a `npfl138.metrics.BIOEncodingF1Score` for constrained decoding and also
            # for greedy decoding, passing both a `list(morpho.train.tags.string_vocab)`
            # and `ignore_index=morpho.PAD`.
            "f1_constrained": npfl138.metrics.BIOEncodingF1Score(
                list(morpho.train.tags.string_vocab),
                ignore_index=MorphoDataset.PAD
            ),
            "f1_greedy": npfl138.metrics.BIOEncodingF1Score(
                list(morpho.train.tags.string_vocab),
                ignore_index=MorphoDataset.PAD
            ),
        },
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
