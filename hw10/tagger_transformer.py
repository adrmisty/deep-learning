#!/usr/bin/env python3
# cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
# fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
# a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.10")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=800, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0.0, type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=0, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=46, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    class FFN(torch.nn.Module):
        def __init__(self, dim: int, expansion: int) -> None:
            super().__init__()
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.ff1 = torch.nn.Linear(dim, dim * expansion)
            self.relu = torch.nn.ReLU()
            self.ff2 = torch.nn.Linear(dim * expansion, dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # TODO: Execute the FFN Transformer layer.
            return self.ff2(self.relu(self.ff1(inputs)))

    class SelfAttention(torch.nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V, and W_O; each a module parameter
            # `torch.nn.Parameter` of shape `[dim, dim]`. The weights should be initialized using
            # the `torch.nn.init.xavier_uniform_` in the same order the matrices are listed above.
            self.W_Q = torch.nn.Parameter(torch.empty(dim, dim))
            self.W_K = torch.nn.Parameter(torch.empty(dim, dim))
            self.W_V = torch.nn.Parameter(torch.empty(dim, dim))
            self.W_O = torch.nn.Parameter(torch.empty(dim, dim))
            for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                torch.nn.init.xavier_uniform_(param)

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `torch.reshape` to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - permute dimensions via `torch.permute` to `[batch_size, heads, max_sentence_len, dim // heads]`.
            B, L, D, H = inputs.shape[0], inputs.shape[1], self.dim, self.heads
            d_k = D // H
            Q = inputs @ self.W_Q  # Shape: [B, L, D]
            K = inputs @ self.W_K  # Shape: [B, L, D]
            V = inputs @ self.W_V  # Shape: [B, L, D]

            Q = Q.reshape(B, L, H, d_k).transpose(1, 2)  # Shape: [B, H, L, d_k]
            K = K.reshape(B, L, H, d_k).transpose(1, 2)  # Shape: [B, H, L, d_k]
            V = V.reshape(B, L, H, d_k).transpose(1, 2)  # Shape: [B, H, L, d_k]

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # Shape: [B, H, L, L]

            # TODO: Apply the softmax, but including a suitable mask ignoring all padding words.
            # The original `mask` is a bool matrix of shape `[batch_size, max_sentence_len]`
            # indicating which words are valid (nonzero value) or padding (zero value).
            # To mask an input to softmax, replace it by -1e9 (theoretically we should use
            # minus infinity, but `torch.exp(-1e9)` is also zero because of limited precision).
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: [B, 1, 1, L]
            scores = scores.masked_fill(~mask, -1e9)
            attention = torch.nn.functional.softmax(scores, dim=-1)  # Shape: [B, H, L, L]

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - permute the result to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - reshape to `[batch_size, max_sentence_len, dim]`,
            # - multiply the result by the W_O matrix.
            x = attention @ V  # Shape: [B, H, L, d_k]
            x = x.transpose(1, 2).reshape(B, L, D)  # Shape: [B, L, D]
            return x @ self.W_O  # Shape: [B, L, D]

    class PositionalEmbedding(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # TODO: Compute the sinusoidal positional embeddings. Assuming the embeddings have
            # a shape `[max_sentence_len, dim]` with `dim` even, and for `0 <= i < dim/2`:
            # - the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / dim))`
            # - the value on index `[pos, dim/2 + i]` should be
            #     `cos(pos / 10_000 ** (2 * i / dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            L, D = inputs.shape[1], inputs.shape[2]
            positions = torch.arange(L, dtype=torch.float32).unsqueeze(1)
            dims = torch.arange(D // 2, dtype=torch.float32).unsqueeze(0)
            angle_rates = 1 / (10000 ** (2 * dims / D))
            angle_rads = positions * angle_rates
            pe = torch.zeros((L, D), dtype=torch.float32, device=inputs.device)
            pe[:, 0:D//2] = torch.sin(angle_rads)
            pe[:, D//2:] = torch.cos(angle_rads)
            return pe.unsqueeze(0).expand(inputs.shape[0], -1, -1)

    class Transformer(torch.nn.Module):
        def __init__(self, layers: int, dim: int, expansion: int, heads: int, dropout: float) -> None:
            super().__init__()
            # TODO: Create:
            # - the positional embedding layer;
            # - the required number of transformer layers, each consisting of
            #   - a layer normalization and a self-attention layer followed by a dropout layer,
            #   - a layer normalization and a FFN layer followed by a dropout layer.
            # During ReCodEx evaluation, the order of layer creation is not important,
            # but if you want to get the same results as on the course website, create
            # the layers in the order they are called in the `forward` method.
            self.positional_embedding = Model.PositionalEmbedding()
            self.layers = torch.nn.ModuleList()
            for _ in range(layers):
                self.layers.append(torch.nn.ModuleDict({
                    "norm1": torch.nn.LayerNorm(dim),
                    "attn": Model.SelfAttention(dim, heads),
                    "drop1": torch.nn.Dropout(dropout),
                    "norm2": torch.nn.LayerNorm(dim),
                    "ffn": Model.FFN(dim, expansion),
                    "drop2": torch.nn.Dropout(dropout)
                }))

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # TODO: First compute the positional embeddings.
            inputs = inputs + self.positional_embedding(inputs)
            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            for layer in self.layers:
                x = layer["norm1"](inputs)
                x = layer["attn"](x, mask)
                x = layer["drop1"](x)
                inputs = inputs + x
                x = layer["norm2"](inputs)
                x = layer["ffn"](x)
                x = layer["drop2"](x)
                inputs = inputs + x
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        # Create all needed layers.
        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = torch.nn.Embedding(len(train.words.string_vocab), args.we_dim)
        # TODO: Create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        self._transformer = Model.Transformer(args.transformer_layers, args.we_dim,
                                              args.transformer_expansion, args.transformer_heads,
                                              args.transformer_dropout)
        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        self._output_layer = torch.nn.Linear(args.we_dim, len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO(tagger_we): Start by embedding the `word_ids` using the word embedding layer.
        hidden = self._word_embedding(word_ids)
        # TODO: Process the embedded words through the transformer. As the second argument,
        # pass the attention mask `word_ids != MorphoDataset.PAD`.
        mask = word_ids != MorphoDataset.PAD
        hidden = self._transformer(hidden, mask)
        # TODO(tagger_we): Pass `hidden` through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimensions.
        hidden = self._output_layer(hidden)
        hidden = hidden.permute(0, 2, 1)
        return hidden


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        forms, tags = example if isinstance(example, tuple) else (example["words"], example["tags"])
        word_ids = torch.tensor([self.dataset.words.string_vocab.index(word) for word in forms], dtype=torch.long)
        tag_ids = torch.tensor([self.dataset.tags.string_vocab.index(tag) for tag in tags], dtype=torch.long)
        return word_ids, tag_ids

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples
        # generated by `transform`.
        word_ids, tag_ids = zip(*batch)
        # TODO(tagger_we): Combine `word_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        # TODO(tagger_we): Process `tag_ids` analogously to `word_ids`.
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(tagger_we): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO(tagger_we): Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation. Note that the loss
        # expects the input to be of shape `[batch_size, num_tags, sequence_length]`.
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),
        # TODO(tagger_we): Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
        # the first argument, `num_classes` set to the number of unique tags, and
        # again `ignore_index=morpho.PAD` to ignore the padded tags.
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass",
                                                     num_classes=len(morpho.train.tags.string_vocab),
                                                     ignore_index=MorphoDataset.PAD)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development and training losses for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if "loss" in metric}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
