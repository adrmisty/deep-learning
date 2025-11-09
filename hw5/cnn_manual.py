#!/usr/bin/env python3
# cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
# fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
# a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
import argparse

import numpy as np
import torch

import npfl138

npfl138.require_version("2425.5")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the implementation.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Convolution:
    def __init__(
            self, filters: int, kernel_size: int, stride: int, input_shape: list[int], verify: bool,
        ) -> None:
        # Create a convolutional layer with the given arguments and given input shape.
        # Note that we use NHWC format, so the MNIST images have shape [28, 28, 1].
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._verify = verify

        # Here the kernel and bias variables are created, the kernel has shape
        # [kernel_height, kernel_width, input_channels, output_channels], bias [output_channels].
        self._kernel = torch.nn.Parameter(torch.randn(kernel_size, kernel_size, input_shape[2], filters) * 0.1)
        self._bias = torch.nn.Parameter(torch.zeros(filters))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, in_channels = inputs.shape
        out_height = (height - self._kernel_size) // self._stride + 1
        out_width = (width - self._kernel_size) // self._stride + 1
        output = torch.zeros(batch_size, out_height, out_width, self._filters, device=inputs.device)

        h_idx_base = torch.arange(out_height, device=inputs.device) * self._stride
        w_idx_base = torch.arange(out_width, device=inputs.device) * self._stride

        for h in range(self._kernel_size):
            for w in range(self._kernel_size):
                h_in_idx = h_idx_base + h
                w_in_idx = w_idx_base + w
                h_in_grid, w_in_grid = torch.meshgrid(h_in_idx, w_in_idx, indexing='ij')

                input_values = inputs[:, h_in_grid, w_in_grid, :]

                input_reshaped = input_values.reshape(batch_size * out_height * out_width, in_channels)
                kernel_weights = self._kernel[h, w, :, :]

                output_reshaped = input_reshaped @ kernel_weights

                output += output_reshaped.reshape(batch_size, out_height, out_width, self._filters)

        output = torch.relu(output + self._bias)

        if self._verify:
            reference = torch.relu(torch.nn.functional.conv2d(
                inputs.movedim(-1, 1), self._kernel.permute(3, 2, 0, 1), self._bias, self._stride)).movedim(1, -1)
            np.testing.assert_allclose(output.detach().numpy(), reference.detach().numpy(), atol=1e-4,
                                       err_msg="Forward pass differs!")

        return output

    def backward(
            self, inputs: torch.Tensor, outputs: torch.Tensor, outputs_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch_size, height, width, in_channels = inputs.shape
        _, out_height, out_width, _ = outputs.shape

        relu_gradient = (outputs > 0) * outputs_gradient
        bias_gradient = relu_gradient.sum(dim=(0, 1, 2))

        h_idx_base = torch.arange(out_height, device=inputs.device) * self._stride
        w_idx_base = torch.arange(out_width, device=inputs.device) * self._stride

        kernel_gradient = torch.zeros_like(self._kernel)

        kernel_np = self._kernel.cpu().numpy()
        relu_gradient_np = relu_gradient.cpu().numpy()
        inputs_gradient_np = np.zeros((batch_size, height, width, in_channels), dtype=np.float32)

        for h in range(self._kernel_size):
            for w in range(self._kernel_size):
                kernel_t = kernel_np[h, w].T

                for i in range(out_height):
                    h_in = i * self._stride + h
                    for j in range(out_width):
                        w_in = j * self._stride + w

                        inputs_gradient_np[:, h_in, w_in, :] += np.dot(relu_gradient_np[:, i, j, :], kernel_t)

        inputs_gradient = torch.tensor(inputs_gradient_np, device=inputs.device)

        for h in range(self._kernel_size):
            for w in range(self._kernel_size):
                h_in_idx = h_idx_base + h
                w_in_idx = w_idx_base + w
                h_in_grid, w_in_grid = torch.meshgrid(h_in_idx, w_in_idx, indexing='ij')

                input_values = inputs[:, h_in_grid, w_in_grid, :]
                input_reshaped = input_values.permute(1, 2, 0, 3).reshape(out_height * out_width * batch_size,
                                                                          in_channels)
                grad_reshaped = relu_gradient.permute(1, 2, 0, 3).reshape(out_height * out_width * batch_size,
                                                                          self._filters)

                kernel_grad = input_reshaped.t() @ grad_reshaped
                kernel_gradient[h, w, :, :] = kernel_grad.reshape(in_channels, self._filters)

        if self._verify:
            with torch.enable_grad():
                inputs.requires_grad_(True)
                inputs.grad = self._kernel.grad = self._bias.grad = None
                reference = (outputs > 0) * torch.nn.functional.conv2d(
                    inputs.movedim(-1, 1), self._kernel.permute(3, 2, 0, 1), self._bias, self._stride).movedim(1, -1)
                reference.backward(gradient=outputs_gradient, inputs=[inputs, self._kernel, self._bias])
                for name, computed, reference in zip(
                        ["Bias", "Kernel", "Inputs"], [bias_gradient, kernel_gradient, inputs_gradient],
                        [self._bias.grad, self._kernel.grad, inputs.grad]):
                    np.testing.assert_allclose(computed.detach().numpy(), reference.detach().numpy(),
                                               atol=2e-4, err_msg=name + " gradient differs!")

        return inputs_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convs = []
        for layer in args.cnn.split(","):
            filters, kernel_size, stride = map(int, layer.split("-"))
            self._convs.append(Convolution(filters, kernel_size, stride, input_shape, args.verify))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, filters]

        # Create the classification head.
        self._classifier = torch.nn.Linear(np.prod(input_shape), MNIST.LABELS)

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            # Forward pass through the convolutions.
            hidden = batch["images"].to(torch.float32).movedim(1, -1) / 255
            conv_values = [hidden]
            for conv in self._convs:
                hidden = conv.forward(hidden)
                conv_values.append(hidden)

            # Run the classification head.
            hidden_flat = torch.flatten(hidden, 1)
            predictions = self._classifier(hidden_flat).softmax(dim=-1)

            # Compute the gradients of the classifier and the convolution output.
            one_hot_labels = torch.nn.functional.one_hot(batch["labels"].to(torch.int64), MNIST.LABELS)
            d_logits = (predictions - one_hot_labels) / len(batch["images"])
            variables = [self._classifier.bias, self._classifier.weight]
            gradients = [d_logits.sum(dim=0), d_logits.T @ hidden_flat]
            hidden_gradient = (d_logits @ self._classifier.weight).reshape(hidden.shape)

            # Backpropagate the gradient through the convolutions.
            for conv, inputs, outputs in reversed(list(zip(self._convs, conv_values[:-1], conv_values[1:]))):
                hidden_gradient, conv_variables, conv_gradients = conv.backward(inputs, outputs, hidden_gradient)
                variables.extend(conv_variables)
                gradients.extend(conv_gradients)

            # Update the weights using a manual SGD.
            for variable, gradient in zip(variables, gradients):
                variable -= self._args.learning_rate * gradient

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        total = correct = 0
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"].to(torch.float32).movedim(1, -1) / 255
            for conv in self._convs:
                hidden = conv.forward(hidden)
            hidden = torch.flatten(hidden, 1)
            predictions = self._classifier(hidden)
            correct += torch.sum(predictions.argmax(dim=-1) == batch["labels"])
            total += len(batch["labels"])
        return correct / total


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Do not compute gradients in this assignment.
    torch.set_grad_enabled(False)

    # Load data, using only 5 000 training images, and create the dataloaders.
    mnist = MNIST(sizes={"train": 5_000})

    # Create the model.
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy))

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy))

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
