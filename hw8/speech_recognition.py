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
import torchaudio.models.decoder
import torchmetrics

import npfl138

npfl138.require_version("2425.8")
from npfl138.datasets.common_voice_cs import CommonVoiceCs
from npfl138 import TrainableModule
from npfl138 import TransformedDataset

# Define parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN cell dimension.")
parser.add_argument("--rnn_layers", default=3, type=int, help="Number of RNN layers.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")
parser.add_argument("--ctc_beam", default=10, type=int, help="Beam size for CTC decoding.")


class Model(TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()

        # 1. CNN Feature Extractor
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(CommonVoiceCs.MFCC_DIM, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )

        # 2. LSTM layers with proper dimensions for bidirectional
        lstm_input_size = 256

        # Create a list to store LSTM layers
        self.lstm_layers = torch.nn.ModuleList()

        # First LSTM layer
        self.lstm_layers.append(
            torch.nn.LSTM(lstm_input_size, args.rnn_dim, bidirectional=True, batch_first=True)
        )

        # Additional LSTM layers - input size is doubled due to bidirectional output
        for _ in range(args.rnn_layers - 1):
            self.lstm_layers.append(
                torch.nn.LSTM(args.rnn_dim * 2, args.rnn_dim, bidirectional=True, batch_first=True)
            )

        # Dropout for regularization
        self.dropout = torch.nn.Dropout(args.dropout)

        # 3. Output projection
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(args.rnn_dim * 2, args.rnn_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.rnn_dim, len(CommonVoiceCs.LETTER_NAMES))
        )

        # CTC decoder
        self._ctc_decoder = torchaudio.models.decoder.ctc_decoder(
            None, CommonVoiceCs.LETTER_NAMES,
            blank_token=CommonVoiceCs.LETTER_NAMES[CommonVoiceCs.PAD],
            sil_token=" ", beam_size=args.ctc_beam, log_add=True
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        torch.nn.init.constant_(param, 0)

    def forward(self, mfccs: torch.Tensor, mfccs_lengths: torch.Tensor) -> torch.Tensor:
        # 1. Apply CNN feature extraction
        x = mfccs.transpose(1, 2)  # [batch, time, features] -> [batch, features, time]
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)  # [batch, features, time] -> [batch, time, features]

        # 2. Apply LSTM layers
        # Pack the sequence for first LSTM
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, mfccs_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process through LSTM layers
        hidden = packed_x
        for i, lstm in enumerate(self.lstm_layers):
            # Process through LSTM
            output, _ = lstm(hidden)

            # Apply dropout to output (except last layer)
            if i < len(self.lstm_layers) - 1:
                # Unpack to apply dropout
                unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                unpacked = self.dropout(unpacked)
                # Repack for next layer
                hidden = torch.nn.utils.rnn.pack_padded_sequence(
                    unpacked, lengths, batch_first=True, enforce_sorted=False
                )
            else:
                hidden = output

        # Unpack final output
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        # 3. Apply final classification layer
        logits = self.classifier(output)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        return log_probs

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs: torch.Tensor,
                     mfccs_lengths: torch.Tensor) -> torch.Tensor:
        # Move tensors to CPU to avoid MPS issues with CTC loss
        y_pred_cpu = y_pred.to('cpu')
        y_true_cpu = y_true.to('cpu')
        mfccs_lengths_cpu = mfccs_lengths.to('cpu')

        # Calculate target lengths (excluding padding)
        target_lengths = torch.sum(y_true_cpu != CommonVoiceCs.PAD, dim=1)

        # Prepare for CTC loss (time-major format)
        log_probs = y_pred_cpu.transpose(0, 1)

        # CTC loss requires non-zero target lengths
        if torch.any(target_lengths == 0):
            target_lengths = torch.clamp(target_lengths, min=1)

        # Calculate CTC loss
        ctc_loss = torch.nn.CTCLoss(blank=CommonVoiceCs.PAD, zero_infinity=True)
        loss = ctc_loss(log_probs, y_true_cpu, mfccs_lengths_cpu, target_lengths)

        # Return loss back to the original device
        return loss.to(y_pred.device)

    def ctc_decoding(self, y_pred: torch.Tensor, mfccs: torch.Tensor, mfccs_lengths: torch.Tensor) -> list[
        torch.Tensor]:
        # Move to CPU for decoding (avoid MPS issues)
        y_pred_cpu = y_pred.to('cpu')
        mfccs_lengths_cpu = mfccs_lengths.to('cpu')

        results = []
        for i in range(y_pred_cpu.size(0)):
            emission = y_pred_cpu[i, :mfccs_lengths_cpu[i]]
            result = self._ctc_decoder(emission.unsqueeze(0))[0][0].tokens
            results.append(torch.tensor(result, device=y_pred.device))

        return results

    def compute_metrics(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, mfccs: torch.Tensor, mfccs_lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Only compute metrics during evaluation
        if self.training:
            return {}

        predictions = self.ctc_decoding(y_pred, mfccs, mfccs_lengths)
        self.metrics["edit_distance"].update(predictions, y_true)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.cpu().numpy() for example in batch]
            return batch

    # Fixed predict_with_dataloader method that handles device placement correctly
    def predict_with_dataloader(self, dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                # Our dataloader returns ((mfccs, mfccs_lengths), targets)
                # We only need the first part for prediction
                inputs, _ = batch

                # Move inputs to the same device as the model
                inputs = tuple(x.to(self.device) for x in inputs)

                # Forward pass
                outputs = self.forward(*inputs)

                # Decode predictions
                batch_predictions = self.ctc_decoding(outputs, *inputs)

                # Convert to numpy and add to list
                predictions.extend([pred.cpu().numpy() for pred in batch_predictions])
        return predictions


class TrainableDataset(TransformedDataset):
    def transform(self, example):
        # Use example["mfccs"] (with an 's') instead of "mfcc"
        mfccs = example["mfccs"].clone().detach().to(torch.float32)

        # Process the target sentence
        target = []
        for char in example["sentence"]:
            if char in CommonVoiceCs.LETTER_NAMES:
                char_idx = CommonVoiceCs.LETTER_NAMES.index(char)
                target.append(char_idx)

        target = torch.tensor(target, dtype=torch.long)
        return mfccs, target

    def collate(self, batch):
        mfccs = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Get lengths for packing
        mfccs_lengths = torch.tensor([mfcc.size(0) for mfcc in mfccs], dtype=torch.long)

        # Pad mfccs to max length
        max_mfcc_len = max(mfcc.size(0) for mfcc in mfccs)
        padded_mfccs = torch.zeros(len(mfccs), max_mfcc_len, mfccs[0].size(1), dtype=torch.float32)
        for i, mfcc in enumerate(mfccs):
            padded_mfccs[i, :mfcc.size(0), :] = mfcc

        # Pad targets to max length
        max_target_len = max(target.size(0) for target in targets)
        padded_targets = torch.zeros(len(targets), max_target_len, dtype=torch.long)
        for i, target in enumerate(targets):
            padded_targets[i, :target.size(0)] = target

        return (padded_mfccs, mfccs_lengths), padded_targets


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    common_voice = CommonVoiceCs()

    # Create datasets
    train = TrainableDataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(common_voice.dev).dataloader(args.batch_size)
    test = TrainableDataset(common_voice.test).dataloader(args.batch_size)

    # Create model and train it
    model = Model(args, common_voice.train)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Configure model
    model.configure(
        optimizer=optimizer,
        loss=torch.nn.CTCLoss(blank=CommonVoiceCs.PAD, zero_infinity=True),
        metrics={
            "edit_distance": common_voice.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD),
        },
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # Use the custom prediction method instead of model.predict()
        predictions = model.predict_with_dataloader(test)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTER_NAMES[char] for char in sentence), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)