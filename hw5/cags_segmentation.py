#!/usr/bin/env python3
#cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
#fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
#a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import npfl138

npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS
from npfl138 import TrainableModule

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
parser.add_argument("--patience", default=5, type=int, help="Early stopping patience")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--model", default="tf_efficientnetv2_b0.in1k", type=str, help="Backbone model from timm")


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()

        self.up_conv1 = nn.ConvTranspose2d(encoder_channels[4], 256, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(encoder_channels[3] + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(encoder_channels[2] + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(encoder_channels[1] + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(encoder_channels[0] + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),  # Added dropout
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up_conv5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, features):
        x = self.up_conv1(features[4])
        x = torch.cat([x, features[3]], dim=1)
        x = self.dec_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.dec_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.dec_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.dec_conv4(x)

        x = self.up_conv5(x)
        x = self.dec_conv5(x)

        return torch.sigmoid(self.final(x))


class SegmentationModel(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model

        dummy_input = torch.zeros(1, 3, 224, 224)
        _, features = self.encoder.forward_intermediates(dummy_input)
        encoder_channels = [f.shape[1] for f in features]

        self.decoder = UNetDecoder(encoder_channels)

    def forward(self, x):
        _, features = self.encoder.forward_intermediates(x)
        return self.decoder(features)


class SegmentationModule(TrainableModule):
    def __init__(self, model):
        super().__init__(model)
        self.mask_iou_metric = CAGS.MaskIoUMetric()

    def compute_metrics(self, y_pred, y, *xs):
        # Update IoU metric
        self.mask_iou_metric.update(y_pred, y)
        return {"iou": self.mask_iou_metric.compute()}


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    cags = CAGS(decode_on_demand=False)

    backbone = timm.create_model(args.model, pretrained=True, num_classes=0)

    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=backbone.pretrained_cfg["mean"],
            std=backbone.pretrained_cfg["std"]
        ),
    ])

    segmentation_model = SegmentationModel(backbone)
    model = SegmentationModule(segmentation_model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=nn.BCELoss(),
        logdir=args.logdir,
        device="auto"
    )

    def collate_fn(batch):
        return (
            torch.stack([preprocessing(item["image"]) for item in batch]),
            torch.stack([item["mask"] for item in batch]),
        )

    train_loader = DataLoader(
        cags.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.threads,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        cags.dev,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.threads,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        cags.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.threads,
        collate_fn=collate_fn
    )

    class EarlyStoppingCallback:
        def __init__(self, patience=5):
            self.patience = patience
            self.best_iou = 0
            self.no_improvement = 0

        def __call__(self, module, epoch, logs):
            iou = logs.get("dev_iou", 0)
            if iou > self.best_iou:
                self.best_iou = iou
                self.no_improvement = 0
                # Save the best model
                os.makedirs(os.path.join(args.logdir, "models"), exist_ok=True)
                torch.save(module.state_dict(), os.path.join(args.logdir, "models", "best_model.pt"))
                print(f"New best IoU: {iou:.4f}")
            else:
                self.no_improvement += 1

            if self.no_improvement >= self.patience:
                print(f"No improvement for {self.patience} epochs, stopping training")
                return TrainableModule.STOP_TRAINING
            return None

    model.fit(
        train_loader,
        epochs=args.epochs,
        dev=dev_loader,
        callbacks=[EarlyStoppingCallback(patience=args.patience)]
    )

    model.load_weights(os.path.join(args.logdir, "models", "best_model.pt"))

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        all_predictions = []
        for batch in test_loader:
            images = batch[0].to(model.device)
            with torch.no_grad():
                predictions = model.module(images)
            all_predictions.extend(predictions.cpu().numpy())

        for mask in all_predictions:
            mask = mask.squeeze()
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)