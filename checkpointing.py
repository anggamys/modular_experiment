import torch
from pathlib import Path
from typing import Optional

from type import LogType
from utils import Utils


class CheckpointManager:
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
        best_path = self.checkpoint_dir / "best_model.pt"
        info_path = self.checkpoint_dir / "best_model_info.json"

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

    def save_last_checkpoint(
        self,
        model,
        optimizer,
        label2id: Optional[dict] = None,
        id2label: Optional[dict] = None,
        char_vocab: Optional[dict] = None,
    ) -> Path:
        last_path = self.checkpoint_dir / "last_model.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label2id": label2id or {},
            "id2label": id2label or {},
            "char_vocab": char_vocab or {},
        }

        torch.save(checkpoint, last_path)

        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Last model saved: {last_path.name}",
        )

        return last_path

    def save_training_results(
        self,
        results: dict,
        label2id: dict,
        id2label: dict,
    ) -> Path:
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
        eval_path = self.checkpoint_dir / "evaluation_results.json"
        self.utils.write_json(eval_path, eval_results)

        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Evaluation results saved: {eval_path.name}",
        )

        return eval_path

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.utils.log(
            "CheckpointManager",
            LogType.INFO,
            f"Checkpoint loaded: {checkpoint_path.name}",
        )

        return checkpoint
