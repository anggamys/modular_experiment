import torch
import torch.nn as nn

from model_base import BaseClassifier


class CharCNNClassifier(BaseClassifier):
    def __init__(
        self,
        num_labels: int,
        char_vocab_size: int,
        char_embedding_dim: int = 64,
        cnn_out_channels: int = 128,
    ):
        super().__init__(num_labels=num_labels)

        self.embedding = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=0
        )

        self.conv = nn.Conv1d(
            char_embedding_dim, cnn_out_channels, kernel_size=3, padding=1
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(cnn_out_channels, num_labels)

    def extract_char_features(self, char_ids):
        x = self.embedding(char_ids).transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)

        return self.dropout(x)

    def forward(self, char_ids=None, labels=None, **kwargs):
        features = self.extract_char_features(char_ids)
        logits = self.classifier(features)

        return self._with_loss(logits, labels)


class CharBiLSTMClassifier(BaseClassifier):
    def __init__(
        self,
        num_labels: int,
        char_vocab_size: int,
        char_embedding_dim: int = 64,
        lstm_hidden_size: int = 128,
    ):
        super().__init__(num_labels=num_labels)

        self.embedding = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)

    def forward(self, char_ids=None, labels=None, **kwargs):
        x = self.embedding(char_ids)

        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        return self._with_loss(logits, labels)


class CharCNNBiLSTMClassifier(BaseClassifier):
    def __init__(
        self,
        num_labels: int,
        char_vocab_size: int,
        char_embedding_dim: int = 64,
        cnn_out_channels: int = 128,
        lstm_hidden_size: int = 128,
    ):
        super().__init__(num_labels=num_labels)

        self.embedding = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=0
        )

        self.conv = nn.Conv1d(
            char_embedding_dim, cnn_out_channels, kernel_size=3, padding=1
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)

    def forward(self, char_ids=None, labels=None, **kwargs):
        x = self.embedding(char_ids).transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2)

        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        return self._with_loss(logits, labels)
