"""
---------------------------------------------------------------------------------

    - Adriana Rodríguez Flórez
    - adrirflorez@gmail.com
    - Course: NPFL138 Deep Learning
    - Homework 3, task "UPPERCASE"
    - Version: ÚFAL 24/25; March 2025 

---------------------------------------------------------------------------------
python uppercase_restoration.py --alphabet_size 150 --embedding_dim 128 --hidden_dim 256 --window 20 --epochs 10 --batch_size 256 --learning_rate 0.001 --output results
>> 
Epoch 1: Dev accuracy [0.9840]. New best model saved! 🎯
Epoch 2: Dev accuracy [0.9863]. New best model saved! 🎯
Epoch 3: Dev accuracy [0.9872]. New best model saved! 🎯
Epoch 4: Dev accuracy [0.9876]. New best model saved! 🎯
Epoch 5: Dev accuracy [0.9877]. New best model saved! 🎯
Epoch 6: Dev accuracy [0.9880]. New best model saved! 🎯
Epoch 7: Dev accuracy [0.9880]. No improvement in (1/7) epochs :()
Epoch 8: Dev accuracy [0.9882]. New best model saved! 🎯
Epoch 9: Dev accuracy [0.9880]. No improvement in (1/7) epochs :()
Epoch 10: Dev accuracy [0.9880]. No improvement in (2/7) epochs :()"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import npfl138
npfl138.require_version("2425.3.1")
from npfl138.datasets.uppercase_data import UppercaseData
from torch.utils.tensorboard.writer import SummaryWriter

# Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", type=int, default=100)
parser.add_argument("--embedding_dim", type=int, default=32)
parser.add_argument("--hidden_dim", type=int, default=256) # 64 -> 256 -> 512
parser.add_argument("--window", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--checkpoint", type=str, default="best_model.pth")


# use this BatchGenerator for the respective train/test/dev data loaders
class BatchGenerator:
    """A simple batch generator, optionally with suffling.

    The functionality of this batch generator is very similar to
        torch.utils.data.DataLoader(
            torch.utils.data.StackDataset(inputs, outputs),
            batch_size=batch_size, shuffle=shuffle,
        )
    but if the data is stored in a single tensor, it is much faster.
    """
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int, shuffle: bool):
        self._inputs = inputs
        self._outputs = outputs
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return (len(self._inputs) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        indices = torch.randperm(len(self._inputs)) if self._shuffle else torch.arange(len(self._inputs))
        while len(indices):
            batch = indices[:self._batch_size]
            indices = indices[self._batch_size:]
            yield self._inputs[batch], self._outputs[batch]

class Model(npfl138.TrainableModule):
    """
    Model: fully connected NN
        - embedding layers (efficient implementation of one-hot encoding followed by a Dense layer).
        - 2 dense layers with ReLU.
        - dropout.
        - normalization.
        - multiclass over characters with cross-entropy loss.

    """
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, window):
        super().__init__()
        self.embedding = nn.Embedding(alphabet_size, embedding_dim)
        # flattening after embedding
        # sliding window of (2*window + 1) characters
        self.fc1 = nn.Linear(embedding_dim * (2 * window + 1), hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, alphabet_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # flatten embedded window to pass to the next layer
        # integrate non-linearity, chosen: ReLU (faster convergence, avoids vanishing gradient problem)
        x = self.embedding(x).view(x.shape[0], -1)
        x = F.relu(self.norm(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm(self.fc2(x)))
        x = self.fc3(x)
        return x

def eval_in_training(model, data):
    """
    Evaluates the (current) model's accuracy during training.
    This is useful for saving the best achieved accuracy during training
    (in a checkpoint path specified by user), as the "best model" so far.
    This best model will be available to be pre-loaded.
    (And baaaasically just for logging and seeing how it is going).
    # useful for saving it when training

    Parameters
    ----------
    model : TrainableModule
        model being trained and subsequently evaluated (in this function)
    data : BatchGenerator
        data loader (as we are in training, train data)
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in data:
            outputs = model(inputs)  # (batch_size, alphabet_size)
            predictions = torch.argmax(outputs, dim=1)  # (batch_size,)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / total


######################## ---------------------------------------------------------
# MODEL TRAINING AND EVALUATION

def train_model(model, train, dev, epochs, learning_rate, check_path, of, stop_th=7):
    """
    Trains model on the Uppercase training data, for the specified hyperparameters
    (epochs and learning rate), saves the best model so far with regards to accuracy at each
    epoch, and documents each accuracy value during training in a given output file.

    Parameters
    ----------
    model + training hyperparameters (train, dev, epochs, learning rate, checkpoint path
    for best trained model, output file)
    stop_th : int
        default value, number of epochs that determine whether there has been a stall in
        accuracy improvement and early stopping should be triggered
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    writer = SummaryWriter(of)
    best_accuracy = 0.0

    # train model for the specified number of epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # keeps track of accuracy of each model trained
        avg_train_loss = total_loss / len(train)
        dev_accuracy = eval_in_training(model, dev)
        scheduler.step(dev_accuracy)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/Dev", dev_accuracy, epoch + 1)
        
        # keep track of the best possible accuracy so far
        # saves model to best model path if this one outperforms the previous
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            stalled = 0
            torch.save(model.state_dict(), check_path)
            print(f"Epoch {epoch+1}: Dev accuracy [{dev_accuracy:.4f}]. New best model saved! 🎯")
        else:
            stalled += 1
            print(f"Epoch {epoch+1}: Dev accuracy [{dev_accuracy:.4f}]. No improvement in ({stalled}/{stop_th}) epochs :()")

        # if no improvement for {stop_th} epochs, stop
        if stalled >= stop_th:
            print("🛑 Early stop due to stall in accuracy improvement! Stopping training... :(")
            break
    
    writer.close()


######################## ---------------------------------------------------------
# MODEL SET UP - TASK SET UP - TASK EXECUTION

def run_uppercase(model, data, output):
    """
    Evaluates trained model and saves outputs
    of model's predictions on uppercase data.

    Parameters
    ----------
    args : program CLI arguments
    
    Returns
    -------
    model : Model

    data : dict
        dictionary containing the entire dataset, and the reespective
        train/dev/test split
    output : string
        path to output file where model outputs will be written to
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data["test"]:
            outputs = model(inputs)
            predictions.extend(torch.argmax(outputs, dim=1).tolist())
    
    with open(output, "w", encoding="utf-8") as f:
        for i, char in enumerate(data["upper"].test.text):
            f.write(char.upper() if predictions[i] else char)

def set_up_model(args, data):
    """
    Creates, builds and trains a given model based on the provided hyperparameters
    and on the respective train and dev datasets.

    Parameters
    ----------
    args : program CLI arguments
    data : dict
        dictionary containing dataset splits [loaders based on BatchGenerator] for the uppercase dataset
    
    Returns
    -------
    model : npfl138.TrainableModule
        model trained on appropriate uppercase data to solve the uppercase NLP task
    """
    # The inputs are for the model are _windows_ of fixed size
    # (`args.window` characters on the left, the character in question, and
    # `args.window` characters on the right)
    model = Model(args.alphabet_size, args.embedding_dim, args.hidden_dim, args.window)
    train_model(model, data["train"], data["dev"], args.epochs, args.learning_rate, args.checkpoint, args.output)
    return model

def set_up_uppercase(args):
    """
    Carries out model creation, training and testing conditions for the uppercase task.
        a. (down)loads input training/testing/dev uppercase data
        b. creates/sets output file for uppercased texts

    Parameters
    ----------
    args : program CLI arguments
    
    Returns
    -------
    data : dict
        dictionary containing the entire dataset, and the reespective
        train/dev/test split
    output : string
        path to output file where model outputs and respective logs will be written to
    gold : Dataset
        gold dataset against which the uppercase task's results will be tested
    """
    # use the npfl138.datasets.uppercase_data module which (down)loads the data
    uppercase_data = UppercaseData(args.window, args.alphabet_size, label_dtype=torch.int64)
    train_data = uppercase_data.train
    dev_data = uppercase_data.dev
    test_data = uppercase_data.test

    # FIX: shuffle=True should be used only for train, not for dev or test
    # the test predictions are then randomly selected from the whole dataset
    # (while you apply them on the text in the original order)
    # therefore by shuffling them it'd be incorrect!!!!!!!!
    data = {"upper": uppercase_data,
            "train": BatchGenerator(train_data.windows, train_data.labels, args.batch_size, shuffle=True),
            "dev": BatchGenerator(dev_data.windows, dev_data.labels, args.batch_size, shuffle=False),
            "test": BatchGenerator(test_data.windows, test_data.labels, args.batch_size, shuffle=False)}
    
    os.makedirs(args.output, exist_ok=True)
    of = os.path.join(args.output, "uppercase_results.txt")

    return data, of


######################## ---------------------------------------------------------
# MAIN + EVALUATION

def main(args):
    # load data + set output data stream
    data, to_output = set_up_uppercase(args)

    # set up model
    model = set_up_model(args, data)
    model.load_state_dict(torch.load(args.checkpoint))
    
    # eval + update best model so far
    eval_in_training(model, data["test"])

    # execute task
    run_uppercase(model, data, to_output)

    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


######################## ---------------------------------------------------------
