import csv

import torch

from collections import Counter
from torch.utils.data import Dataset


class JokeDataset(Dataset):

    def __init__(
        self,
        file_path,
        seq_length=3
    ):

        self.seq_length = seq_length

        jokes = []

        # -----------------------------------------
        # Read CSV
        # -----------------------------------------

        with open(
            file_path,
            'r',
            encoding='utf-8'
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                joke = row['Joke'].lower()

                jokes.append(joke)

        # -----------------------------------------
        # Combine all jokes into one text
        # -----------------------------------------

        text = ' '.join(jokes)

        # tokenize

        self.words = text.split()

        # -----------------------------------------
        # Build vocabulary
        # -----------------------------------------

        counter = Counter(self.words)

        self.vocab = {
            word: i
            for i, word in enumerate(counter.keys())
        }

        self.idx_to_word = {
            i: word
            for word, i in self.vocab.items()
        }

        # -----------------------------------------
        # Encode words
        # -----------------------------------------

        self.encoded = [
            self.vocab[word]
            for word in self.words
        ]

    # -----------------------------------------

    def __len__(self):

        return len(self.encoded) - self.seq_length

    # -----------------------------------------

    def __getitem__(self, idx):

        x = self.encoded[
            idx:idx+self.seq_length
        ]

        y = self.encoded[
            idx+self.seq_length
        ]

        return (
            torch.tensor(x),
            torch.tensor(y)
        )