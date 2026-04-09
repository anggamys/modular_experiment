"""
Model builder that orchestrates model construction.
"""

from typing import Optional

from transformers import PreTrainedModel

from bert_models import (
    BertCNNClassifier,
    BertGRUClassifier,
    BertLinearClassifier,
    BertMLPClassifier,
)
from char_models import (
    CharBiLSTMClassifier,
    CharCNNBiLSTMClassifier,
    CharCNNClassifier,
)
from hybrid_models import HybridBertCharCNNClassifier
from type import LogType
from utils import Utils


class ModelBuilder:
    """Builds model instances based on configuration."""

    def __init__(self):
        self.utils = Utils()

    @staticmethod
    def uses_transformer(architecture: str) -> bool:
        """Check if architecture uses transformer models."""
        return architecture in {
            "bert_linear",
            "bert_mlp",
            "bert_gru",
            "bert_cnn",
            "hybrid_bert_charcnn",
        }

    def _build_bert_backbone(self, bert_model, freeze_bert):
        """Build BERT backbone with optional freezing."""
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
        """
        Build model instance based on configuration.

        Args:
            config_model: Model configuration dict
            num_labels: Number of classification labels
            bert_model: Pre-trained BERT model (required for BERT architectures)
            char_vocab_size: Character vocabulary size (required for char/hybrid architectures)

        Returns:
            Constructed model instance
        """
        architecture = config_model["architecture"]

        # Validate requirements
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

        # Build model based on architecture
        if architecture == "bert_linear":
            model = BertLinearClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model.get("hidden_size", 768),
            )
        elif architecture == "bert_mlp":
            model = BertMLPClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model.get("hidden_size", 768),
                mlp_hidden_size=config_model.get("mlp_hidden_size", 256),
            )
        elif architecture == "bert_gru":
            model = BertGRUClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model.get("hidden_size", 768),
                gru_hidden_size=config_model.get("gru_hidden_size", 256),
            )
        elif architecture == "bert_cnn":
            model = BertCNNClassifier(
                model=self._build_bert_backbone(
                    bert_model, config_model["freeze_bert"]
                ),
                num_labels=num_labels,
                hidden_size=config_model.get("hidden_size", 768),
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
                hidden_size=config_model.get("hidden_size", 768),
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
