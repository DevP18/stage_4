import torch
import torch.nn as nn


class RNNGenerator(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        rnn_type='RNN'
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        if rnn_type == 'RNN':

            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                batch_first=True
            )

        elif rnn_type == 'LSTM':

            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                batch_first=True
            )

        elif rnn_type == 'GRU':

            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                batch_first=True
            )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

        self.rnn_type = rnn_type

    # -----------------------------------------

    def forward(self, x):

        embedded = self.embedding(x)

        output, hidden = self.rnn(embedded)

        if self.rnn_type == 'LSTM':

            hidden = hidden[0]

        hidden = hidden[-1]

        out = self.fc(hidden)

        return out