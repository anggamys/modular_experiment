import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel

from type import LogType
from utils import Utils


class IndoBERTForTokenClassification(nn.Module):
    def __init__(self, model: PreTrainedModel, num_labels: int, hidden_size: int = 768):
        super().__init__()
        self.utils = Utils()
        self.bert = model
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(hidden_size, num_labels)

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

        outputs = {"logits": logits}

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

        return bert_output.last_hidden_state[:, 0, :]

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
