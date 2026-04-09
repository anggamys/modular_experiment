"""
Training policy enforcement module.

Handles epoch policies and learning rate policies to ensure
training configurations are realistic per architecture.
"""

from type import LogType
from utils import Utils


class TrainingPolicyManager:
    """Manages training policies (epochs, learning rates) per architecture."""

    EPOCH_POLICY = {
        "bert_linear": (3, 5),
        "bert_mlp": (3, 5),
        "bert_gru": (4, 6),
        "bert_cnn": (4, 6),
        "char_cnn": (8, 15),
        "char_bilstm": (8, 15),
        "char_cnn_bilstm": (10, 20),
        "hybrid_bert_charcnn": (4, 8),
    }

    LR_POLICY = {
        "char_cnn": 1e-3,
        "char_bilstm": 7.5e-4,
        "char_cnn_bilstm": 5e-4,
        "_default_char": 5e-4,
        "_default_bert": 2e-5,
        "_default_distilbert": 3e-5,
    }

    def __init__(self):
        self.utils = Utils()

    def apply_epoch_policy(self, config: dict, enforce_cap: bool = False) -> dict:
        """
        Enforce epoch policy based on architecture.

        Args:
            config: Training config dict
            enforce_cap: If True, also enforce max epoch cap

        Returns:
            Updated config dict
        """
        training_cfg = config.get("training", {})
        if not training_cfg.get("enforce_epoch_policy", True):
            return config

        architecture = str(config.get("model", {}).get("architecture", "")).lower()
        min_epoch, max_epoch = self.EPOCH_POLICY.get(architecture, (3, 5))

        current_epochs = int(training_cfg.get("num_epochs", max_epoch))
        if enforce_cap:
            adjusted_epochs = max(min_epoch, min(current_epochs, max_epoch))
        else:
            adjusted_epochs = max(min_epoch, current_epochs)

        if adjusted_epochs != current_epochs:
            self.utils.log(
                "TrainingPolicyManager",
                LogType.WARNING,
                f"num_epochs={current_epochs} adjusted to {adjusted_epochs} "
                f"for architecture={architecture}",
            )

        config["training"]["num_epochs"] = adjusted_epochs
        return config

    def apply_learning_rate_policy(self, config: dict) -> dict:
        """
        Enforce learning rate policy based on architecture.

        Args:
            config: Training config dict

        Returns:
            Updated config dict with recommended LR
        """
        training_cfg = config.get("training", {})
        if not training_cfg.get("enforce_learning_rate_policy", True):
            return config

        architecture = str(config.get("model", {}).get("architecture", "")).lower()
        model_name = str(config.get("model", {}).get("model_name", "")).lower()

        recommended_lr = self._compute_recommended_lr(architecture, model_name)
        current_lr = float(training_cfg.get("learning_rate", recommended_lr))

        if current_lr != recommended_lr:
            self.utils.log(
                "TrainingPolicyManager",
                LogType.WARNING,
                f"learning_rate={current_lr} adjusted to {recommended_lr} "
                f"for architecture={architecture}",
            )

        config["training"]["learning_rate"] = recommended_lr

        # Set defaults for split learning rates
        config["training"].setdefault("bert_learning_rate", 2e-5)
        config["training"].setdefault("head_learning_rate", 1e-4)
        config["training"].setdefault("char_learning_rate", 1e-3)

        return config

    def _compute_recommended_lr(self, architecture: str, model_name: str) -> float:
        """Compute recommended learning rate for given architecture."""
        if architecture.startswith("char_"):
            return self.LR_POLICY.get(architecture, self.LR_POLICY["_default_char"])

        if architecture == "hybrid_bert_charcnn":
            return (
                self.LR_POLICY["_default_distilbert"]
                if "distilbert" in model_name
                else self.LR_POLICY["_default_bert"]
            )

        # Default BERT/Hybrid
        return (
            self.LR_POLICY["_default_distilbert"]
            if "distilbert" in model_name
            else self.LR_POLICY["_default_bert"]
        )
