import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional
from transformers import PreTrainedModel

from type import LogType
from utils import Utils


class BaseClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.utils = Utils()
        self.num_labels = num_labels
        self.loss_fn = CrossEntropyLoss()

    def _with_loss(self, logits, labels=None):
        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)
        return outputs


class BertLinearClassifier(BaseClassifier):
    def __init__(self, model: PreTrainedModel, num_labels: int, hidden_size: int = 768):
        super().__init__(num_labels=num_labels)
        self.bert = model
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.utils.log(
            "BertLinearClassifier",
            LogType.INFO,
            f"Model initialized with num_labels={num_labels}, hidden_size={hidden_size}",
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
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


class ModelBuilder:
    def __init__(self):
        self.utils = Utils()

    def _build_bert_backbone(self, bert_model, freeze_bert):
        if bert_model is None:
            self.utils.log("ModelBuilder", LogType.ERROR, "BERT model is required.")
            exit(1)

        for param in bert_model.parameters():
            param.requires_grad = not freeze_bert

        return bert_model

    def build_model(
        self,
        config_model: dict,
        num_labels: int,
        bert_model: Optional[PreTrainedModel] = None,
        char_vocab_size: Optional[int] = None,
    ):
        architecture = config_model["architecture"]

        if (
            architecture
            in {
                "char_cnn",
                "char_bilstm",
                "char_cnn_bilstm",
                "hybrid_bert_charcnn",
            }
            and char_vocab_size is None
        ):
            self.utils.log(
                "ModelBuilder",
                LogType.ERROR,
                "char_vocab_size is required for char/hybrid architectures.",
            )
            exit(1)

        char_vocab_size_safe = (
            int(char_vocab_size) if char_vocab_size is not None else 0
        )

        if architecture == "bert_linear":
            model = BertLinearClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model["hidden_size"],
            )
        elif architecture == "bert_mlp":
            model = BertMLPClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model["hidden_size"],
                mlp_hidden_size=config_model.get("mlp_hidden_size", 256),
            )
        elif architecture == "bert_gru":
            model = BertGRUClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model["hidden_size"],
                gru_hidden_size=config_model.get("gru_hidden_size", 256),
            )
        elif architecture == "bert_cnn":
            model = BertCNNClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model["hidden_size"],
                cnn_out_channels=config_model.get("cnn_out_channels", 128),
            )
        elif architecture == "char_cnn":
            model = CharCNNClassifier(
                num_labels=num_labels,
                char_vocab_size=char_vocab_size_safe,
                char_embedding_dim=config_model.get("char_embedding_dim", 64),
                cnn_out_channels=config_model.get("cnn_out_channels", 128),
            )
        elif architecture == "char_bilstm":
            model = CharBiLSTMClassifier(
                num_labels=num_labels,
                char_vocab_size=char_vocab_size_safe,
                char_embedding_dim=config_model.get("char_embedding_dim", 64),
                lstm_hidden_size=config_model.get("lstm_hidden_size", 128),
            )
        elif architecture == "char_cnn_bilstm":
            model = CharCNNBiLSTMClassifier(
                num_labels=num_labels,
                char_vocab_size=char_vocab_size_safe,
                char_embedding_dim=config_model.get("char_embedding_dim", 64),
                cnn_out_channels=config_model.get("cnn_out_channels", 128),
                lstm_hidden_size=config_model.get("lstm_hidden_size", 128),
            )
        elif architecture == "hybrid_bert_charcnn":
            model = HybridBertCharCNNClassifier(
                bert_model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model["hidden_size"],
                char_vocab_size=char_vocab_size_safe,
                char_embedding_dim=config_model.get("char_embedding_dim", 64),
                cnn_out_channels=config_model.get("cnn_out_channels", 128),
            )
        else:
            self.utils.log(
                "ModelBuilder",
                LogType.ERROR,
                f"Unsupported architecture: {architecture}",
            )
            exit(1)

        self.utils.log(
            "ModelBuilder",
            LogType.INFO,
            f"Model ready: architecture={architecture}, num_labels={num_labels}",
        )

        return model


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
