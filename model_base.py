"""
Base classifier and model utilities.
"""

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel

from utils import Utils


class BaseClassifier(nn.Module):
    """Base class for all classifiers."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.utils = Utils()
        self.num_labels = num_labels
        self.loss_fn = CrossEntropyLoss()

    def _with_loss(self, logits, labels=None):
        """Combine logits with loss computation if labels provided."""
        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)
        return outputs


def infer_bert_hidden_size(model: PreTrainedModel, fallback: int = 768) -> int:
    """Infer hidden size from BERT model config."""
    config_hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if config_hidden_size is not None:
        return int(config_hidden_size)
    return int(fallback)
