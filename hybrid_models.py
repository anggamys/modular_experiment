import torch
import torch.nn as nn
from transformers import PreTrainedModel

from char_models import CharCNNClassifier
from model_base import BaseClassifier, infer_bert_hidden_size


class HybridBertCharCNNClassifier(BaseClassifier):
    def __init__(
        self,
        bert_model: PreTrainedModel,
        num_labels: int,
        hidden_size: int,
        char_vocab_size: int,
        char_embedding_dim: int = 64,
        cnn_out_channels: int = 128,
    ):
        super().__init__(num_labels=num_labels)

        self.bert = bert_model
        hidden_size = infer_bert_hidden_size(bert_model, hidden_size)

        self.char_encoder = CharCNNClassifier(
            num_labels=num_labels,
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            cnn_out_channels=cnn_out_channels,
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size + cnn_out_channels, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        char_ids=None,
        labels=None,
        **kwargs,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        bert_pooled = bert_output.last_hidden_state[:, 0, :]
        char_features = self.char_encoder.extract_char_features(char_ids)
        features = torch.cat([bert_pooled, char_features], dim=1)
        features = self.dropout(features)
        logits = self.classifier(features)

        return self._with_loss(logits, labels)
