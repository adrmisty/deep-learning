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
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision.ops import batched_nms
import torchvision.transforms.v2 as v2
from torchmetrics import Accuracy, MeanSquaredError
from torch.utils.data import DataLoader

import bboxes_utils
import npfl138

import warnings

warnings.filterwarnings("ignore")

npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN
from npfl138 import TrainableModule
from npfl138 import TransformedDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.001, type=float, help="Maximum number of threads to use.")

NUM_ANCHORS = 196
IMAGE_SIZE = 224
ANCHOR_SIZE = 16



def classification_subnet():
    class ClassificationSubnet(nn.Module):
        def __init__(self):
            super(ClassificationSubnet, self).__init__()

            layers = []
        
            for _ in range(3):
                layers.extend([
                    nn.Conv2d(112, 112, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(112),
                    nn.ReLU()
                ])

            layers.append(nn.Conv2d(112, SVHN.LABELS, kernel_size=3, stride=1, padding=1))

            self.features = nn.Sequential(*layers)

        def forward(self, x):
            x = self.features(x)

            batch_size, channels, height, width = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(batch_size, NUM_ANCHORS, SVHN.LABELS)

            return x

    return ClassificationSubnet()


def bbox_subnet():
    class BboxSubnet(nn.Module):
        def __init__(self):
            super(BboxSubnet, self).__init__()

            layers = []
      
            for _ in range(3):
                layers.extend([
                    nn.Conv2d(112, 112, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(112),
                    nn.ReLU()
                ])

            layers.append(nn.Conv2d(112, 4, kernel_size=3, stride=1, padding=1))

            self.features = nn.Sequential(*layers)

        def forward(self, x):
            x = self.features(x)

            batch_size, channels, height, width = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(batch_size, NUM_ANCHORS, 4)

            return x

    return BboxSubnet()


class SVHNDetector(TrainableModule):
    def __init__(self, backbone, feature_level=3):
        super(SVHNDetector, self).__init__()
        self.backbone = backbone
        self.feature_level = feature_level
        self.classification_subnet = classification_subnet()
        self.bbox_subnet = bbox_subnet()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        output, features = self.backbone.forward_intermediates(x)
        feature = features[self.feature_level]

        classification_output = self.classification_subnet(feature)
        bbox_output = self.bbox_subnet(feature)

        return classification_output, bbox_output

    def compute_loss(self, y_pred, y, *xs):
        classification_pred, bbox_pred = y_pred

        classification_target = y[0]
        bbox_target = y[1]
        bbox_weights = y[2]
        bbox_weights = bbox_weights.bool()

        classification_loss = torchvision.ops.sigmoid_focal_loss(classification_pred, classification_target, reduction='mean')

        bbox_pred, bbox_target = bbox_pred[bbox_weights], bbox_target[bbox_weights]
        bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_target, reduction='sum') / bbox_pred.shape[0]

        total_loss = classification_loss + bbox_loss

        return total_loss


def get_anchors(image_size, anchor_size):
    coordinates = []
    for y in range(0, image_size, anchor_size):
        for x in range(0, image_size, anchor_size):
            coordinates.append([y, x, y + anchor_size, x + anchor_size])

    return coordinates


anchors = torch.tensor(get_anchors(IMAGE_SIZE, ANCHOR_SIZE))


def resize_image(image) -> torch.Tensor:
    image = torchvision.transforms.functional.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image


back = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

transforming = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
    v2.Normalize(mean=back.pretrained_cfg["mean"], std=back.pretrained_cfg["std"]),
])

def prepare_example(example, is_training=True):
    orig_height, orig_width = example["image"].shape[1], example["image"].shape[2]

    image = example["image"]
    resized_image = torchvision.transforms.functional.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    resized_image = transforming(resized_image)
    bboxes = example["bboxes"] / np.array([orig_height, orig_width, orig_height, orig_width]) * IMAGE_SIZE

    anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(anchors, example["classes"], bboxes, iou_threshold=0.5)

    resized_image_tensor = resized_image.float()
    anchor_classes_tensor = torch.tensor(anchor_classes, dtype=torch.long)
    anchor_bboxes_tensor = torch.tensor(anchor_bboxes, dtype=torch.float32)

    # removing the column with background
    one_hot = F.one_hot(anchor_classes_tensor, num_classes=SVHN.LABELS+1).float()[:,1:]
    positive_mask = (anchor_classes_tensor > 0).float()

    return (
        resized_image_tensor,
        (
            one_hot,
            anchor_bboxes_tensor,
            positive_mask
        )
    )


def prepare_test_example(example):
    resized_image = resize_image(example["image"])
    return resized_image



def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    svhn = SVHN(decode_on_demand=False)

    train_dataset = TransformedDataset(svhn.train)
    dev_dataset = TransformedDataset(svhn.dev)
    test_dataset = TransformedDataset(svhn.test)

    train_dataset.transform = prepare_example
    dev_dataset.transform = prepare_example
    test_dataset.transform = prepare_test_example

    NUM_WORKERS = 6
    train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    dev = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS)
    test = DataLoader(svhn.test, batch_size=args.batch_size, num_workers=NUM_WORKERS)

    backbone = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=backbone.pretrained_cfg["mean"], std=backbone.pretrained_cfg["std"]),
    ])

    model = SVHNDetector(backbone)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.configure(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epochs * len(train),
            eta_min=0),
        logdir=args.logdir,
        device="auto"
    )

    # Train the model
    model.fit(
        train,
        epochs=args.epochs,
        dev=dev
    )


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Get predictions for the test set
        for example in svhn.test:
            # Process a single example at a time
            image = example["image"]
            resized_img = resize_image(image)
            processed_img = preprocessing(resized_img.unsqueeze(0)).to(model.device)

            with torch.no_grad():
                class_outputs, bbox_outputs = model(processed_img)

            class_outputs_cpu = class_outputs.cpu().squeeze()
            # print(class_outputs_cpu.shape)
            # class_outputs_sigmoid = torch.sigmoid(class_outputs_cpu.reshape(196*10)).reshape(196, 10)
            class_outputs_sigmoid = torch.sigmoid(class_outputs_cpu)

            class_outputs_np = class_outputs_sigmoid.numpy()
            bbox_outputs_cpu = bbox_outputs.cpu()

            bboxes = bboxes_utils.bboxes_from_rcnn(anchors, bbox_outputs_cpu[0])

            bboxes_np = bboxes.numpy()
            # print(class_outputs_np[0])
            # print(class_outputs_np[1])
            output = []
            categories = [np.argmax(classes) for classes in class_outputs_np]
            scores = [classes[np.argmax(classes)] for classes in class_outputs_np]

            idxs = batched_nms(
                bboxes,
                torch.tensor(scores),
                torch.tensor(categories),
                0.5
            ).numpy()

            original_image_height = example["image"].shape[1]
            original_image_width = example["image"].shape[2]
            original_image_size = np.array(
                [original_image_height, original_image_width, original_image_height, original_image_width]
            )

            for idx in idxs:
                label = np.argmax(class_outputs_np[idx])
                if class_outputs_np[idx, label] > 0.25:
                    resized_bbox = bboxes_np[idx] / IMAGE_SIZE * original_image_size
                    output += [label] + list(resized_bbox)

            print(*output, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
