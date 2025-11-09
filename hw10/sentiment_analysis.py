#!/usr/bin/env python3
# cc1d5497-f262-4cfd-b538-a12d2b403848 - Rodríguez Flórez Adriana
# fc0aa991-f366-44b1-992d-2c70e9f7a995 - Ahmadi AbdulAli
# a436e2ea-25cf-4f36-8ee9-d81ff6c9f813 - Korol Andrii
#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import transformers
from torch.utils.data import DataLoader

import npfl138
npfl138.require_version("2425.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--max_length", default=128, type=int, help="Maximum sequence length.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset) -> None:
        super().__init__()
        self.eleczech = eleczech
        self.num_labels = len(dataset.label_vocab)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(eleczech.config.hidden_size, self.num_labels)

    # TODO: Implement the model computation.
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        outputs = self.eleczech(input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state has shape [batch_size, sequence_length, hidden_size]
        # We take the embedding of the first token (index 0) as the representation of the whole sequence
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        encoded = self.tokenizer(example["document"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        label_str = example["label"]
        try:
            label_index = self.dataset.label_vocab._string_map[label_str]
        except KeyError:
            # Handle the case where the label is not in the vocabulary
            # You might want to assign a special index for unknown labels (if your vocab has UNK)
            # Or, if it's an empty string issue, you might want to handle it specifically.
            # For simplicity, let's assign a default value (e.g., 0) or raise an error with more context.
            #print(f"Warning: Label '{label_str}' not found in vocabulary. Assigning index 0.")
            label_index = 0  # Or handle differently based on your needs

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": label_index,
        }

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        labels = torch.tensor([example["label"] for example in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    
    
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

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    train_dataset = TrainableDataset(facebook.train, tokenizer, args.max_length)
    dev_dataset = TrainableDataset(facebook.dev, tokenizer, args.max_length)
    test_dataset = TrainableDataset(facebook.test, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dev_dataset.collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate)

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model and move it to the device.
    model = Model(args, eleczech, facebook.train).to(device)

    # TODO: Configure and train the model
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(facebook.train.label_vocab)).to(device)

    def train_epoch(epoch):
        model.train()
        total_loss = 0
        #for batch in train_loader:
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs)
            loss = loss_fn(logits, labels)
            if i % 50 ==0:
                print(f'{i+1} / {len(train_loader)} batch loss: {loss}')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_metric(torch.tensor(all_preds), torch.tensor(all_labels))
        return total_loss / len(loader), accuracy.item()

    print(f"Training for {args.epochs} epochs.")
    for epoch in range(args.epochs):
        train_loss = train_epoch(epoch + 1)
        dev_loss, dev_accuracy = evaluate(dev_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Dev Loss = {dev_loss:.4f}, Dev Accuracy = {dev_accuracy:.4f}")
        
        if (epoch+1) % 2==0:
            # Generate test set annotations, but in `args.logdir` to allow parallel execution.
            os.makedirs(args.logdir, exist_ok=True)
            predictions_path = os.path.join(args.logdir, f"sentiment_analysis{epoch}.txt")
            with open(predictions_path, "w", encoding="utf-8") as predictions_file:
                # TODO: Predict the tags on the test set.
                print("start predictiong for test file...")
                model.eval()
                for batch in test_loader:
                    inputs = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                    }
                    with torch.no_grad():
                        logits = model(**inputs)
                    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                    for prediction in predictions:
                        print(facebook.train.label_vocab.string(prediction), file=predictions_file)

            print(f"Test set annotations saved to: {predictions_path}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)