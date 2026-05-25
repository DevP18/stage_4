import torch
import torch.nn as nn

from stage_4_code.method import method


class RNNClassifier(method, nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=2,
        rnn_type='RNN'
    ):

        method.__init__(
            self,
            "RNNClassifier",
            "RNN text classifier"
        )

        nn.Module.__init__(self)

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
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.rnn_type = rnn_type

    def forward(self, x):
        

        embedded = self.embedding(x)

        output, hidden = self.rnn(embedded)

        if self.rnn_type == 'LSTM':

            hidden = hidden[0]

        hidden = hidden[-1]

        hidden = self.dropout(hidden)

        out = self.fc(hidden)

        return out

    def run(self, trainData, trainLabel, testData):

        return self.forward(testData)