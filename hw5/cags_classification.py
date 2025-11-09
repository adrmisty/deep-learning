#!/usr/bin/env python3
# cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
# fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
# a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
import argparse
import datetime
import os
import re
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
import npfl138

npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS
from npfl138.trainable_module import TrainableModule

# *TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")


class CAGSClassifier(TrainableModule):
    def __init__(self, backbone, num_classes, dropout=0.2):
        model = nn.Sequential(
            backbone,
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )
        super().__init__(model)


class TorchAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("correct", torch.tensor(0, dtype=torch.int64), persistent=False)
        self.register_buffer("total", torch.tensor(0, dtype=torch.int64), persistent=False)

    def reset(self):
        self.correct.zero_()
        self.total.zero_()

    def update(self, y_pred, y_true):
        _, predicted = torch.max(y_pred, 1)
        self.correct += (predicted == y_true).sum().item()
        self.total += y_true.size(0)

    def compute(self):
        return 100.0 * self.correct / self.total


class CAGSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocessing):
        self.dataset = dataset
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = self.preprocessing(example["image"])
        return image, example["label"]

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

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # *TODO: Create the model and train it.
    train_dataset = CAGSDataset(cags.train, preprocessing)
    val_dataset = CAGSDataset(cags.dev, preprocessing)
    test_dataset = CAGSDataset(cags.test, preprocessing)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size
    )
    test = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size
    )

    model = CAGSClassifier(
        backbone=efficientnetv2_b0,
        num_classes=CAGS.LABELS,
        dropout=args.dropout
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={"accuracy": TorchAccuracy()},
        logdir=args.logdir,
        device="auto"
    )

    model.fit(train_loader, epochs=args.epochs, dev=val_loader)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # *TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)