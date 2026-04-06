import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel

from type import LogType
from utils import Utils


# Architecture: IndoBERT → Linear → Softmax
class IndoBERTForTokenClassification(nn.Module):
    def __init__(self, model: PreTrainedModel, num_labels: int, hidden_size: int = 768):
        super().__init__()
        self.utils = Utils()
        self.bert = model
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Linear layer for classification
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Loss function
        self.loss_fn = CrossEntropyLoss()

        self.utils.log(
            "IndoBERTForTokenClassification",
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
        # Get BERT output
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # Use [CLS] token representation or mean pooling of all tokens
        # For single token, we use the token representation at position
        hidden_states = (
            bert_output.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # For single token classification, use the first token representation
        # or you can use mean pooling
        pooled_output = hidden_states[
            :, 0, :
        ]  # Use [CLS] token: (batch_size, hidden_size)

        # Classification layer
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        outputs = {"logits": logits}

        # Calculate loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs["loss"] = loss

        return outputs

    def get_embedding_features(self, input_ids, attention_mask=None):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return bert_output.last_hidden_state[:, 0, :]  # [CLS] token

    def freeze_bert_encoder(self, freeze: bool = True):
        for param in self.bert.parameters():
            param.requires_grad = not freeze

        status = "frozen" if freeze else "unfrozen"
        self.utils.log(
            "IndoBERTForTokenClassification",
            LogType.INFO,
            f"BERT encoder {status}",
        )

    def unfreeze_bert_encoder(self):
        self.freeze_bert_encoder(False)


# Architecture: IndoBERT → Linear → CRF
class IndoBERTForTokenClassificationWithCRF(IndoBERTForTokenClassification):
    def __init__(self, model: PreTrainedModel, num_labels: int, hidden_size: int = 768):
        super().__init__(model, num_labels, hidden_size)


# Architecture: IndoBERT → BiLSTM → Softmax
class IndoBERTForTokenClassificationWithBiLSTM(nn.Module):
    def __init__(self, model: PreTrainedModel, num_labels: int, hidden_size: int = 768):
        super().__init__()
        self.utils = Utils()
        self.bert = model
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Linear layer for classification
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Loss function
        self.loss_fn = CrossEntropyLoss()

        self.utils.log(
            "IndoBERTForTokenClassificationWithBiLSTM",
            LogType.INFO,
            f"Model initialized with num_labels={num_labels}, hidden_size={hidden_size}",
        )
