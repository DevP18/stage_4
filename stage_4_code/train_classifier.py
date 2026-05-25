import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from stage_4_code.text_dataset import TextDataset
from stage_4_code.rnn_classifier import RNNClassifier

from stage_4_code.config import *


def train_model(rnn_type='RNN'):

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------

    dataset = TextDataset(
        'stage_4_data/text_classification/train'
    )

    test_dataset = TextDataset(
        'stage_4_data/text_classification/test',
        vocab=dataset.vocab
    )

    # -------------------------------------------------
    # Create dataloaders
    # -------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------

    model = RNNClassifier(
        vocab_size=len(dataset.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        rnn_type=rnn_type
    )

    model.to(DEVICE)

    # -------------------------------------------------
    # Loss and optimizer
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    losses = []

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for x, y in dataloader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        losses.append(avg_loss)

        print(
            f'Epoch {epoch+1}, Loss: {avg_loss:.4f}'
        )

    # -------------------------------------------------
    # Evaluate on test dataset
    # -------------------------------------------------

    correct = 0

    total = 0

    model.eval()

    with torch.no_grad():

        for x, y in test_dataloader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            output = model(x)

            predictions = torch.argmax(
                output,
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

    accuracy = correct / total

    print(
        f'{rnn_type} Test Accuracy: {accuracy:.4f}'
    )

    # -------------------------------------------------
    # Save learning curve
    # -------------------------------------------------

    os.makedirs(
        'stage_4_code/results',
        exist_ok=True
    )

    plt.plot(losses)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title(f'{rnn_type} Learning Curve')

    plt.savefig(
        f'stage_4_code/results/{rnn_type}_curve.png'
    )

    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == '__main__':

    train_model('RNN')

    train_model('LSTM')

    train_model('GRU')