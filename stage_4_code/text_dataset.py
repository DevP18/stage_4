import os
from collections import Counter

import torch
from torch.utils.data import Dataset

from stage_4_code.dataset import dataset


class TextDataset(dataset, Dataset):

    def __init__(
        self,
        root_dir,
        seq_length=200,
        vocab=None
    ):

        super().__init__(
            "TextDataset",
            "IMDB Text Classification Dataset"
        )

        self.root_dir = root_dir

        self.seq_length = seq_length

        self.texts = []

        self.labels = []

        self.load_files()
        if vocab is None:
            self.build_vocab()
        else:
            self.vocab = vocab
        

    # -------------------------------------------------
    # Load all text files
    # -------------------------------------------------

    def load_files(self):

        # positive reviews

        pos_dir = os.path.join(
            self.root_dir,
            'pos'
        )

        for filename in os.listdir(pos_dir):

            file_path = os.path.join(
                pos_dir,
                filename
            )

            with open(
                file_path,
                'r',
                encoding='utf-8'
            ) as f:

                text = f.read()

            self.texts.append(text)

            self.labels.append(1)

        # negative reviews

        neg_dir = os.path.join(
            self.root_dir,
            'neg'
        )

        for filename in os.listdir(neg_dir):

            file_path = os.path.join(
                neg_dir,
                filename
            )

            with open(
                file_path,
                'r',
                encoding='utf-8'
            ) as f:

                text = f.read()

            self.texts.append(text)

            self.labels.append(0)

    # -------------------------------------------------
    # Build vocabulary
    # -------------------------------------------------

    def build_vocab(self):

        all_words = []

        for text in self.texts:

            words = text.lower().split()

            all_words.extend(words)

        counter = Counter(all_words)

        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1
        }

        for word in counter:

            self.vocab[word] = len(self.vocab)

    # -------------------------------------------------
    # Convert text -> numbers
    # -------------------------------------------------

    def encode_text(self, text):

        words = text.lower().split()

        encoded = [

            self.vocab.get(word, 1)

            for word in words
        ]

        # padding

        if len(encoded) < self.seq_length:

            encoded += [0] * (
                self.seq_length - len(encoded)
            )

        # truncate

        else:

            encoded = encoded[:self.seq_length]

        return torch.tensor(encoded)

    # -------------------------------------------------

    def __len__(self):

        return len(self.texts)

    # -------------------------------------------------

    def __getitem__(self, idx):

        x = self.encode_text(
            self.texts[idx]
        )

        y = torch.tensor(
            self.labels[idx]
        )

        return x, y

    # -------------------------------------------------

    def load(self):

        return self