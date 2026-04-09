"""
Checkpoint and artifact management module.

Handles saving/loading checkpoints, best model tracking, and result artifacts.
"""

from pathlib import Path
from typing import Optional

from type import LogType
from utils import Utils


class CheckpointManager:
    """Manages model checkpoints and training artifacts."""

    def __init__(self, checkpoint_dir: Path, utils: Utils):
        self.checkpoint_dir = checkpoint_dir
        self.utils = utils
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model,
        optimizer,
        monitor_value: Optional[float] = None,
        monitor_name: Optional[str] = None,
        label2id: Optional[dict] = None,
        id2label: Optional[dict] = None,
        char_vocab: Optional[dict] = None,
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            model: PyTorch model
            optimizer: Optimizer
            monitor_value: Value of monitored metric
            monitor_name: Name of monitored metric
            label2id: Label to ID mapping
            id2label: ID to label mapping
            char_vocab: Character vocabulary (if applicable)

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "monitor": monitor_name,
            "monitor_value": monitor_value,
            "label2id": label2id or {},
            "id2label": id2label or {},
            "char_vocab": char_vocab or {},
        }

        import torch

        torch.save(checkpoint, checkpoint_path)

        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Checkpoint saved: {checkpoint_path.name}",
        )
        return checkpoint_path

    def save_best_checkpoint(
        self,
        model,
        optimizer,
        monitor_value: Optional[float] = None,
        monitor_name: Optional[str] = None,
        label2id: Optional[dict] = None,
        id2label: Optional[dict] = None,
        char_vocab: Optional[dict] = None,
    ) -> Path:
        """
        Save best model checkpoint as 'best_model.pt' and metadata.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            monitor_value: Value of monitored metric
            monitor_name: Name of monitored metric
            label2id: Label to ID mapping
            id2label: ID to label mapping
            char_vocab: Character vocabulary

        Returns:
            Path to best_model.pt
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        info_path = self.checkpoint_dir / "best_model_info.json"

        import torch

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "monitor": monitor_name,
            "monitor_value": monitor_value,
            "label2id": label2id or {},
            "id2label": id2label or {},
            "char_vocab": char_vocab or {},
        }

        torch.save(checkpoint, best_path)

        self.utils.write_json(
            info_path,
            {
                "monitor": monitor_name,
                "monitor_value": float(monitor_value) if monitor_value else None,
                "saved_at": self.utils.dateTimeNow(),
            },
        )

        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Best model saved: {best_path.name} (monitor={monitor_name}, value={monitor_value})",
        )
        return best_path

    def save_training_results(
        self,
        results: dict,
        label2id: dict,
        id2label: dict,
    ) -> Path:
        """
        Save training results as JSON.

        Args:
            results: Training results dict
            label2id: Label to ID mapping
            id2label: ID to label mapping

        Returns:
            Path to results file
        """
        results_path = self.checkpoint_dir / "training_results.json"

        results_with_labels = {
            **results,
            "label2id": label2id,
            "id2label": id2label,
        }

        self.utils.write_json(results_path, results_with_labels)

        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Training results saved: {results_path.name}",
        )
        return results_path

    def save_evaluation_results(self, eval_results: dict) -> Path:
        """
        Save evaluation results as JSON.

        Args:
            eval_results: Evaluation results dict

        Returns:
            Path to evaluation results file
        """
        eval_path = self.checkpoint_dir / "evaluation_results.json"
        self.utils.write_json(eval_path, eval_results)

        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Evaluation results saved: {eval_path.name}",
        )
        return eval_path

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Loaded checkpoint dict
        """
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Checkpoint loaded: {checkpoint_path.name}",
        )
        return checkpoint
