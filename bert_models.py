import torch
import torch.nn as nn
from transformers import PreTrainedModel

from model_base import BaseClassifier, infer_bert_hidden_size
from type import LogType


class BertLinearClassifier(BaseClassifier):
    def __init__(self, model: PreTrainedModel, num_labels: int, hidden_size: int = 768):
        super().__init__(num_labels=num_labels)

        self.bert = model
        self.hidden_size = infer_bert_hidden_size(model, hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        self.utils.log(
            "BertLinearClassifier",
            LogType.INFO,
            f"Model initialized with num_labels={num_labels}, hidden_size={self.hidden_size}",
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        hidden_states = bert_output.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        pooled_output = hidden_states[:, 0, :]
        logits = self.classifier(pooled_output)

        return self._with_loss(logits, labels)

    def freeze_bert_encoder(self, freeze: bool = True):
        for param in self.bert.parameters():
            param.requires_grad = not freeze

        status = "frozen" if freeze else "unfrozen"

        self.utils.log(
            "BertLinearClassifier",
            LogType.INFO,
            f"BERT encoder {status}",
        )

    def unfreeze_bert_encoder(self):
        self.freeze_bert_encoder(False)


class BertMLPClassifier(BaseClassifier):
    def __init__(
        self,
        model: PreTrainedModel,
        num_labels: int,
        hidden_size: int = 768,
        mlp_hidden_size: int = 256,
    ):
        super().__init__(num_labels=num_labels)

        self.bert = model
        hidden_size = infer_bert_hidden_size(model, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        pooled_output = self.dropout(bert_output.last_hidden_state[:, 0, :])
        logits = self.classifier(pooled_output)

        return self._with_loss(logits, labels)


class BertGRUClassifier(BaseClassifier):
    def __init__(
        self,
        model: PreTrainedModel,
        num_labels: int,
        hidden_size: int = 768,
        gru_hidden_size: int = 256,
    ):
        super().__init__(num_labels=num_labels)

        self.bert = model
        hidden_size = infer_bert_hidden_size(model, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(gru_hidden_size * 2, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = bert_output.last_hidden_state
        _, hidden = self.gru(sequence_output)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        return self._with_loss(logits, labels)


class BertCNNClassifier(BaseClassifier):
    def __init__(
        self,
        model: PreTrainedModel,
        num_labels: int,
        hidden_size: int = 768,
        cnn_out_channels: int = 128,
    ):
        super().__init__(num_labels=num_labels)

        self.bert = model
        hidden_size = infer_bert_hidden_size(model, hidden_size)
        self.conv = nn.Conv1d(hidden_size, cnn_out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(cnn_out_channels, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = bert_output.last_hidden_state.transpose(1, 2)
        cnn_features = torch.relu(self.conv(sequence_output))
        pooled = self.pool(cnn_features).squeeze(-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return self._with_loss(logits, labels)
