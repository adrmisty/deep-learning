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
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

class ResidualBlock(npfl138.TrainableModule):
    """
    Auxiliary class implementing the residual connections.
    """
    def __init__(self, inside_layers):
        super().__init__() # gets inside layers as parameter
                            # in order to have the whole network in a single torch sequential
        self.layers = torch.nn.Sequential(*inside_layers)
        
    def forward(self, x):
        return x + self.layers(x)


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255.0  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


class Model(npfl138.TrainableModule):

    ## AUX functions
    # repeated in several parts of the code

    def parse_cnn_layers(self, definition : str):
        """
        Splits a CNN definition string, taking into account:
         - comma separated values
          -bracket separated values
        saving the latter as they will be part of the future residual layers!."""
        
        layers = []
        aux = ""
        # nested brackets inside the cnn def
        level = 0 

        for char in definition:
            if char == "," and level == 0:
                layers.append(aux); aux = ""
            else:
                aux += char
                if char == "[":
                    level += 1
                elif char == "]":
                    level -= 1

        if aux:
            layers.append(aux)

        return layers


    def parse_cnn_params(self, cnn_parts : list):
        """
        Parses the list of parameters for CNN creation.
        """
        filters = int(cnn_parts[1])
        kernel_size = int(cnn_parts[2])
        stride = int(cnn_parts[3])
        padding = cnn_parts[4]# no need to calc padding!! can be same, valid or a str

        return filters, kernel_size, stride, padding

    def C_layer(self, in_channels, filters, kernel, stride, padding):
        """ 
        Creates a convolutional layer followed by ReLU. 
        """
        return [
            torch.nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel, stride=stride, padding=padding),
            torch.nn.ReLU()
        ]

    def CB_layer(self, in_channels, filters, kernel, stride, padding):
        """ 
        Adds a convolutional layer without bias, followed by batch normalization and ReLU. 
        """
        return [
            torch.nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU()
        ]


    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes the Convolutional Network model by adding CNN layers specified by `args.cnn`, which contains
        a comma-separated list of the layers specified in the comments.
        """
        super().__init__()
        layers = []
        in_channels = MNIST.C  # initialize MNIST's input channels (1 for grayscale), these will be updated with each layer
        cnn_layers = self.parse_cnn_layers(args.cnn) if args.cnn else []

        for l in cnn_layers:
            parts = l.split("-")
            layer = parts[0]
        
            # -------- CONVOLUTIONAL LAYER
            # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
            #   activation and specified number of filters, kernel size, stride and padding.
            if layer == "C":
                filters, kernel, stride, padding = self.parse_cnn_params(parts)
                layers.extend(self.C_layer(in_channels, filters, kernel, stride, padding))
                in_channels = filters
            
            # -------- CONVOLUTIONAL LAYER with BATCH NORMALIZATION
            # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
            #   In detail, start with a convolutional layer **without bias** and activation,
            #   then add a batch normalization layer, and finally the ReLU activation.
            elif layer == "CB":
                filters, kernel, stride, padding = self.parse_cnn_params(parts)
                layers.extend(self.CB_layer(in_channels, filters, kernel, stride, padding))
                in_channels = filters
            
            # -------- MAX POOLING
            # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
            #   the default padding of 0 (the "valid" padding).
            elif layer == "M": 
                pool_size, stride = map(int, parts[1:])
                layers.append(torch.nn.MaxPool2d(pool_size, stride))

            # -------- RESIDUAL LAYERS
            # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
            #   of at least one convolutional layer (but not a recursive residual connection `R`).
            #   The input to the `R` layer should be processed sequentially by `layers`, and the
            #   produced output (after the ReLU nonlinearity of the last layer) should be added
            #   to the input (of this `R` layer).  
            elif layer == "R":
                residual_layers = []

                # parse the sublayers recursively
                # [...]
                res_block_spec = l[l.index("[") + 1 : l.rindex("]")] 
                res_layers = self.parse_cnn_layers(res_block_spec)

                for res_spec in res_layers:
                    res_parts = res_spec.split("-")
                    res_type = res_parts[0]

                    # residual convolutional layer
                    if res_type == "C":
                        filters, kernel, stride, padding = self.parse_cnn_params(res_parts)
                        residual_layers.extend(self.C_layer(in_channels, filters, kernel, stride, padding))
                        in_channels = filters

                    # residual convolutional layer with batch norm
                    elif res_type == "CB":
                        filters, kernel, stride, padding = self.parse_cnn_params(res_parts)
                        residual_layers.extend(self.CB_layer(in_channels, filters, kernel, stride, padding))
                        in_channels = filters

                layers.append(ResidualBlock(residual_layers))
                
            # -------- FLATTEN
            # - `F`: Flatten inputs. 
            elif layer == "F":
                layers.append(torch.nn.Flatten())
                in_channels = None  # must appear exactly once in the architecture, transition to dense layers
                
            # It might be difficult to compute the number of features after the `F` layer. You can
            # nevertheless use the `torch.nn.LazyLinear` and `torch.nn.LazyConv2d` layers, which
            # do not require the number of input features to be specified in the constructor.
            
            # -------- HIDDEN LAYER
            # - `H-hidden_layer_size`: Add a (fully connected hidden) dense layer with ReLU activation and the specified size.
            elif layer == "H":
                hidden_size = int(parts[1])
                layers.append(torch.nn.LazyLinear(hidden_size))
                layers.append(torch.nn.ReLU())
                
            # - `D-dropout_rate`: Apply dropout with the given dropout rate.
            # You can assume the resulting network is valid; it is fine to crash if it is not.
            elif layer == "D":
                dropout_rate = float(parts[1])
                layers.append(torch.nn.Dropout(dropout_rate))
                
        # TODO-solved: Finally, add the final Linear output layer with `MNIST.LABELS` units.
        # final classification layer
        layers.append(torch.nn.LazyLinear(MNIST.LABELS))
        self.model = torch.nn.Sequential(*layers)
        
        # However, after the whole model is constructed, you must call the model once on a dummy input
        # so that the number of features is computed and the model parameters are initialized.
        # To that end, you can use for example
        #   self.eval()(torch.zeros(1, MNIST.C, MNIST.H, MNIST.W))
        # where the `self.eval()` is necessary to avoid the batchnorms to update their running statistics.
        self.eval()(torch.zeros(1, MNIST.C, MNIST.H, MNIST.W))

    def forward(self, x):
        return self.model(x)


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

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it
    model = Model(args)

    model.train() # avoid batch freezing
    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)