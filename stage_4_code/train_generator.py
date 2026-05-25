import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from stage_4_code.joke_dataset import JokeDataset
from stage_4_code.rnn_generator import RNNGenerator

from stage_4_code.config import *


# -------------------------------------------------

def generate_text(
    model,
    dataset,
    start_words,
    num_words=30
):

    model.eval()

    words = start_words.lower().split()

    for _ in range(num_words):

        x = [
            dataset.vocab.get(word, 0)
            for word in words[-3:]
        ]

        x = torch.tensor(x).unsqueeze(0)

        x = x.to(DEVICE)

        with torch.no_grad():

            output = model(x)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

        next_word = dataset.idx_to_word[
            prediction
        ]

        words.append(next_word)

    return ' '.join(words)


# -------------------------------------------------

def train_model(rnn_type='RNN'):

    dataset = JokeDataset(
        'stage_4_data/text_generation/data'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = RNNGenerator(
        vocab_size=len(dataset.vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        rnn_type=rnn_type
    )

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------
    # Store losses for learning curve
    # -----------------------------------------

    losses = []

    # -----------------------------------------
    # Training loop
    # -----------------------------------------

    for epoch in range(EPOCHS):

        total_loss = 0

        model.train()

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

        # -----------------------------------------
        # Save epoch loss
        # -----------------------------------------

        losses.append(avg_loss)

        print(
            f'Epoch {epoch+1}, Loss: {avg_loss:.4f}'
        )

    # -----------------------------------------
    # Plot learning curve
    # -----------------------------------------

    plt.plot(losses)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title(
        f'{rnn_type} Generation Loss Curve'
    )

    plt.savefig(
        f'stage_4_code/results/{rnn_type}_generation_curve.png'
    )

    # -----------------------------------------
    # Generate text
    # -----------------------------------------

    print('\nGenerated Joke:\n')

    start_words = input('Enter 3 starting words: ')

    generated = generate_text(model,dataset,start_words,30)

    print(generated)


# -------------------------------------------------

if __name__ == '__main__':

    train_model('RNN')

    train_model('LSTM')

    train_model('GRU')