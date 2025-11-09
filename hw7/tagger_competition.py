#!/usr/bin/env python3
# cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
# fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
# a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
import argparse
import datetime
import os
import re

import numpy as np
import torch
import npfl138

npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset

import fasttext
import fasttext.util

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=96, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=96, type=int, help="RNN layer dimension.")
parser.add_argument("--char_dim", default=96, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=96, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.1, type=float, help="Mask words with the given probability.")
parser.add_argument("--word_rnn_layers", default=1, type=int, help="Number of word rnn layers.")
parser.add_argument("--char_rnn_layers", default=1, type=int, help="Number of char rnn layers.")
parser.add_argument("--mlp_size", default=96, type=int, help="Number of char rnn layers.")


class Model(npfl138.TrainableModule):
    class MaskElements(torch.nn.Module):
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self._mask_probability:
                mask = torch.rand_like(inputs, dtype=torch.float32) < self._mask_probability
                inputs = torch.where(mask, torch.tensor(self._mask_value, device=inputs.device), inputs)
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset, fasttext_model=None) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._fasttext_model = fasttext_model
        self._word_masking = self.MaskElements(args.word_masking, train.words.string_vocab.index(MorphoDataset.UNK))
        self._char_embedding = torch.nn.Embedding(len(train.words.char_vocab), args.cle_dim)
        self._char_rnn = torch.nn.GRU(args.cle_dim, args.cle_dim, bidirectional=True, batch_first=True)
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)

        # Initialize with FastText vectors if available
        if fasttext_model is not None:
            with torch.no_grad():
                for i in range(len(train.words.string_vocab)):
                    try:
                        word = train.words.string_vocab.string(i)
                        if word != MorphoDataset.PAD and word != MorphoDataset.UNK:
                            vector = torch.tensor(fasttext_model.get_word_vector(word), dtype=torch.float)
                            # Ensure vector size matches embedding dimension
                            if vector.shape[0] > args.we_dim:
                                vector = vector[:args.we_dim]
                            elif vector.shape[0] < args.we_dim:
                                # Pad with zeros if needed
                                temp_vector = torch.zeros(args.we_dim, dtype=torch.float)
                                temp_vector[:vector.shape[0]] = vector
                                vector = temp_vector
                            self._word_embedding.weight[i] = vector
                    except Exception as e:
                        pass  # Skip any issues

        rnn_cls = torch.nn.LSTM if args.rnn == "LSTM" else torch.nn.GRU
        self._word_rnn = rnn_cls(args.we_dim + 2 * args.cle_dim, args.rnn_dim, bidirectional=True, batch_first=True)
        self._output_layer = torch.nn.Linear(2 * args.rnn_dim, len(train.tags.string_vocab))

        # Move model to GPU if available
        self.to(self.device)

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device as the model
        word_ids = word_ids.to(self.device)
        unique_words = unique_words.to(self.device)
        word_indices = word_indices.to(self.device)

        hidden = self._word_masking(word_ids)
        word_embedded = self._word_embedding(hidden)
        cle_embedded = self._char_embedding(unique_words)

        # Move lengths tensor to CPU as PyTorch requires
        packed = torch.nn.utils.rnn.pack_padded_sequence(cle_embedded,
                                                         lengths=(unique_words != MorphoDataset.PAD).sum(dim=1).cpu(),
                                                         batch_first=True, enforce_sorted=False)
        _, states = self._char_rnn(packed)

        if isinstance(states, tuple):
            states = states[0]

        cle = torch.cat([states[-2], states[-1]], dim=1)
        cle = torch.nn.functional.embedding(word_indices, cle)

        hidden = torch.cat([word_embedded, cle], dim=2)

        # Move lengths tensor to CPU as PyTorch requires
        sentence_lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, sentence_lengths, batch_first=True,
                                                         enforce_sorted=False)
        packed_output, _ = self._word_rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # For bidirectional RNN, combine forward and backward outputs
        forward = hidden[:, :, :self._word_rnn.hidden_size]
        backward = hidden[:, :, self._word_rnn.hidden_size:]
        hidden = torch.cat([forward, backward], dim=2)

        hidden = self._output_layer(hidden)
        hidden = hidden.permute(0, 2, 1)

        return hidden


class MorphoTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        word_ids = torch.tensor([self.dataset.words.string_vocab.index(word) for word in example['words']])
        tag_ids = torch.tensor([self.dataset.tags.string_vocab.index(tag) for tag in example['tags']])
        return word_ids, example["words"], tag_ids


def main(args: argparse.Namespace) -> None:
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    morpho = MorphoDataset("czech_pdt")

    # Check if the model file exists before downloading
    model_path = 'cc.cs.300.bin'
    if not os.path.exists(model_path):
        print(f"Downloading FastText model for Czech...")
        fasttext.util.download_model('cs', if_exists='ignore')

    try:
        ft = fasttext.load_model(model_path)
        print("FastText model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load FastText model: {e}")
        print("Continuing without FastText embeddings")
        ft = None

    model = Model(args, morpho.train, ft)

    train = MorphoTaggingDataset(morpho.train)
    dev = MorphoTaggingDataset(morpho.dev)
    test = MorphoTaggingDataset(morpho.test)

    def prepare_batch(data):
        word_ids, words, tag_ids = zip(*data)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        unique_words, word_indices = morpho.train.cle_batch(words)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True)
        return (word_ids, unique_words, word_indices), tag_ids

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch,
                                               shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, collate_fn=prepare_batch)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, collate_fn=prepare_batch)

    # Define metrics (empty dict as in the original code)
    metrics = {}

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5),
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics=metrics,
        logdir=args.logdir)

    logs = model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set annotations
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Manually predict on the test set without using model.predict()
        model.eval()
        test_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, _ = batch
                # Forward pass with unpacked tuple - no need to move inputs to device as it's handled in forward method
                outputs = model(*inputs)
                predicted_indices = outputs.argmax(dim=1).cpu().numpy()

                # Get word lengths to crop predictions
                word_lengths = (inputs[0] != morpho.PAD).sum(dim=1).cpu().numpy()

                # Store predictions with their corresponding words
                for i, (pred, length) in enumerate(zip(predicted_indices, word_lengths)):
                    test_predictions.append(pred[:length])

        # Write predictions to file
        for i, (predicted_indices, words) in enumerate(zip(test_predictions, morpho.test.words.strings)):
            for idx in predicted_indices:
                try:
                    tag = morpho.train.tags.string_vocab.string(idx)
                    print(tag, file=predictions_file)
                except IndexError:
                    # Handle potential index errors by printing a default tag
                    print("X", file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)