#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# Create a dataset from consecutive _pairs_ of original examples, assuming
# that the size of the original dataset is even.
class DatasetOfPairs(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self._dataset = dataset

    def __len__(self):
        # TODO-solved: The new dataset has half the size of the original one.
        return len(self._dataset) // 2

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        # TODO-solved: Given an `index`, generate an example composed of two input examples.
        # Notably, considering examples `self._dataset[2 * index]` and `self._dataset[2 * index + 1]`,
        # each being a dictionary with keys "image" and "label", return a pair `(input, output)` with
        # - `input` being a pair of images, each converted to `torch.float32` and divided by 255,
        # - `output` being a pair of labels.
        a = self._dataset[2 * index]
        b = self._dataset[2 * index + 1]

        image_a = (a["image"].to(torch.float32) / 255.0); label_a = a["label"]
        image_b = (b["image"].to(torch.float32) / 255.0); label_b = b["label"]

        # just like in the previous exercise
        return (image_a, image_b), (label_a, label_b)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO-solved: Create all layers required to implement the forward pass.
        
        # shared by image 1 and image 2
        self.feature_extractor = torch.nn.Sequential(
            # CL1: convolution with 10 filters, 3x3 kernel, stride 2, valid padding, ReLU
            torch.nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=0), 
            torch.nn.ReLU(),
            # CL2: convolution with 20 filters, 3x3 ernel, stride 2, valid padding, ReLU
            torch.nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            # flattening layer
            torch.nn.Flatten(),
            # LL: fully connected for 200dim image vectors
            # feature size: 720
            torch.nn.Linear(20 * 6 * 6, 200),
            torch.nn.ReLU()
        )

        self.compare_layer = torch.nn.Sequential(
            # LL: linear with 200 neurons for concatenation of the 2x200 image feat vectors
            torch.nn.Linear(2 * 200, 200),
            torch.nn.ReLU(),
            # LL: output linear with sigmoid activation
            torch.nn.Linear(200, 1),
            torch.nn.Sigmoid()
        )

        # classification
        # LL: linear layer with 10 classes for the 200dim image feature vectors
        self.digit_classifier = torch.nn.Linear(200, 10)


    def forward(
        self, first: torch.Tensor, second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO-solved: Implement the forward pass of the model using the layers created in the constructor.
        
        # The model starts by passing each input image through the same
        # module (with shared weights), for feature extraction
        first_feat = self.feature_extractor(first)
        second_feat = self.feature_extractor(second)
        to_compare = torch.cat([first_feat, second_feat], dim=1)

        # 1. first, compute _direct comparison_ whether the first digit is
        #   greater than the second, by
        direct_comparison = self.compare_layer(to_compare)

        # 2. classify the computed representation FV of the first image using
        #   a linear layer into 10 classes; do the sam for the second
        digit_1 = self.digit_classifier(first_feat)
        digit_2 = self.digit_classifier(second_feat)

        # 3. finally, compute _indirect comparison_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        indirect_comparison = (digit_1.argmax(dim=1) > digit_2.argmax(dim=1)).float().unsqueeze(1)

        return direct_comparison, digit_1, digit_2, indirect_comparison

    def compute_loss(self, y_pred, y_true, *inputs):
        # The `compute_loss` method can override the loss computation of the model.
        # It is needed when there are multiple model outputs or multiple losses to compute.
        # We start by unpacking the multiple outputs of the model and the multiple targets.
        direct_comparison_pred, digit_1_pred, digit_2_pred, indirect_comparison_pred = y_pred
        digit_1_true, digit_2_true = y_true
        
        # TODO-solved: Compute the required losses. Note that the `direct_comparison_pred` is
        # really a probability (sigmoid was applied), while the `digit_1_pred` and
        # `digit_2_pred` are logits of 10-class classification.
        direct_comparison_true = (digit_1_true > digit_2_true).float()
        direct_comparison_loss = torch.nn.functional.binary_cross_entropy(
            direct_comparison_pred.squeeze(), direct_comparison_true
        )
        digit_1_loss = torch.nn.functional.cross_entropy(digit_1_pred, digit_1_true)
        digit_2_loss = torch.nn.functional.cross_entropy(digit_2_pred, digit_2_true)

        return direct_comparison_loss + digit_1_loss + digit_2_loss

    def compute_metrics(self, y_pred, y_true, *inputs):
        # The `compute_metrics` can override metric computation for the model. We start by
        # unpacking the multiple outputs of the model and the multiple targets.
        direct_comparison_pred, digit_1_pred, digit_2_pred, indirect_comparison_pred = y_pred
        digit_1_true, digit_2_true = y_true

        # TODO-solved: Update two metrics -- the `direct_comparison` and the `indirect_comparison`.
        direct_comparison_true = (digit_1_true > digit_2_true).float()
        indirect_comparison_true = (digit_1_true > digit_2_true).float()

        self.metrics["direct_comparison"].update(direct_comparison_pred.squeeze(), direct_comparison_true)
        self.metrics["indirect_comparison"].update(indirect_comparison_pred.squeeze(), indirect_comparison_true)

        # Finally, we return the dictionary of all the metric values.
        return {name: metric.compute() for name, metric in self.metrics.items()}


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

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(DatasetOfPairs(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(DatasetOfPairs(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it
    model = Model(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        metrics={
            # TODO-solved: Create two binary accuracy metrics using `torchmetrics.Accuracy`:
            "direct_comparison": torchmetrics.Accuracy(task="binary"),
            "indirect_comparison": torchmetrics.Accuracy(task="binary"),
        },
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)