"""
Main training orchestrator module.

Coordinates training workflow using Policy, Checkpoint, and Evaluator modules.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpointing import CheckpointManager
from evaluator import Evaluator
from training_policy import TrainingPolicyManager
from type import LogType
from utils import Utils


class Trainer:
    """Main trainer class orchestrating the training workflow."""

    def __init__(self, config_path: str, exp_id: str | None = None):
        self.utils = Utils()

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Apply training policies
        policy_manager = TrainingPolicyManager()
        raw_config = policy_manager.apply_epoch_policy(raw_config)
        raw_config = policy_manager.apply_learning_rate_policy(raw_config)

        self.config = self._resolve_experiment_config(raw_config, exp_id)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.config["training"]["device"] == "cuda"
            else "cpu"
        )

        self._set_seed(self.config["experiment"]["seed"])

        save_dir = Path(self.config["output"]["model_save_dir"])
        exp_name = self.config["experiment"].get(
            "code", self.config["experiment"]["name"]
        )
        self.checkpoint_dir = save_dir / exp_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint and evaluator managers
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, self.utils)
        self.evaluator = Evaluator(self.checkpoint_dir, self.utils, self.device)

        # Always keep an experiment-local log in checkpoint dir
        self.utils.log2file(log_dir=str(self.checkpoint_dir), filename="run.log")

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Device: {self.device} | Path: {self.checkpoint_dir}",
        )

    def _resolve_experiment_config(self, raw_config: dict, exp_id: str | None):
        """Resolve and merge experiment-specific configuration."""
        experiments = raw_config.get("experiments", {})
        if exp_id is None:
            exp_id = raw_config.get("experiment", {}).get("default_id")

        if exp_id and exp_id in experiments:
            selected = experiments[exp_id]
            merged = {
                "model": {**raw_config.get("model", {}), **selected.get("model", {})},
                "training": {
                    **raw_config.get("training", {}),
                    **selected.get("training", {}),
                },
                "data": {**raw_config.get("data", {}), **selected.get("data", {})},
                "output": {
                    **raw_config.get("output", {}),
                    **selected.get("output", {}),
                },
                "experiment": {
                    **raw_config.get("experiment", {}),
                    **selected.get("experiment", {}),
                    "code": exp_id,
                },
            }
            self.utils.log("Trainer", LogType.INFO, f"Selected experiment: {exp_id}")
            return merged

        return raw_config

    def _build_optimizer(self, model) -> tuple:
        """Build optimizer with optional split learning rates."""
        training_cfg = self.config["training"]
        architecture = str(self.config["model"].get("architecture", "")).lower()
        use_split_lr = bool(training_cfg.get("use_split_learning_rate", False))
        weight_decay = float(training_cfg["weight_decay"])

        base_lr = float(training_cfg["learning_rate"])
        bert_lr = float(training_cfg.get("bert_learning_rate", base_lr))
        head_lr = float(training_cfg.get("head_learning_rate", base_lr))
        char_lr = float(training_cfg.get("char_learning_rate", base_lr))

        if architecture.startswith("char_"):
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=char_lr,
                weight_decay=weight_decay,
            )
            self.utils.log("Trainer", LogType.INFO, f"Optimizer LR (char): {char_lr}")
            return optimizer, {
                "strategy": "char-global",
                "effective_param_group_lrs": [{"name": "char", "lr": char_lr}],
            }

        if use_split_lr and hasattr(model, "bert"):
            bert_params = [p for p in model.bert.parameters() if p.requires_grad]
            bert_param_ids = {id(p) for p in bert_params}

            if architecture == "hybrid_bert_charcnn" and hasattr(model, "char_encoder"):
                char_params = [
                    p
                    for p in model.char_encoder.parameters()
                    if p.requires_grad and id(p) not in bert_param_ids
                ]
                char_param_ids = {id(p) for p in char_params}
                head_params = [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                    and id(p) not in bert_param_ids
                    and id(p) not in char_param_ids
                ]

                param_groups = [
                    {"params": bert_params, "lr": bert_lr},
                    {"params": char_params, "lr": char_lr},
                    {"params": head_params, "lr": head_lr},
                ]

                optimizer = AdamW(param_groups, weight_decay=weight_decay)
                self.utils.log(
                    "Trainer",
                    LogType.INFO,
                    f"Split optimizer (hybrid): bert_lr={bert_lr}, char_lr={char_lr}, head_lr={head_lr}",
                )
                return optimizer, {
                    "strategy": "split-hybrid",
                    "effective_param_group_lrs": [
                        {"name": "bert", "lr": bert_lr},
                        {"name": "char", "lr": char_lr},
                        {"name": "head", "lr": head_lr},
                    ],
                }

            else:
                head_params = [
                    p
                    for p in model.parameters()
                    if p.requires_grad and id(p) not in bert_param_ids
                ]

                param_groups = [
                    {"params": bert_params, "lr": bert_lr},
                    {"params": head_params, "lr": head_lr},
                ]

                optimizer = AdamW(param_groups, weight_decay=weight_decay)
                self.utils.log(
                    "Trainer",
                    LogType.INFO,
                    f"Split optimizer (bert-head): bert_lr={bert_lr}, head_lr={head_lr}",
                )
                return optimizer, {
                    "strategy": "split-bert-head",
                    "effective_param_group_lrs": [
                        {"name": "bert", "lr": bert_lr},
                        {"name": "head", "lr": head_lr},
                    ],
                }

        # Global optimizer
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            weight_decay=weight_decay,
        )
        self.utils.log("Trainer", LogType.INFO, f"Global optimizer LR: {base_lr}")
        return optimizer, {
            "strategy": "global",
            "effective_param_group_lrs": [{"name": "global", "lr": base_lr}],
        }

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _move_batch_to_device(self, batch: dict):
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _create_dataloaders(self, train_dataset, val_dataset):
        """Create train and validation dataloaders."""
        batch_size = self.config["training"]["batch_size"]
        return (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=self.device.type == "cuda",
            ),
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
            ),
        )

    def _build_scheduler(self, optimizer, train_loader):
        """Build learning rate scheduler."""
        num_training_steps = len(train_loader) * self.config["training"]["num_epochs"]
        return LinearLR(optimizer, start_factor=1.0, total_iters=num_training_steps)

    def _resolve_early_stopping_config(self) -> tuple:
        """Resolve early stopping configuration."""
        training_cfg = self.config["training"]
        monitor = training_cfg.get("early_stopping_monitor", "val_loss")
        patience = int(training_cfg.get("early_stopping_patience", 2))
        return monitor, patience

    def _init_training_results(self, optimizer_lr_details: dict) -> dict:
        """Initialize training results dict."""
        return {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "metrics": [],
            "epoch_train_seconds": [],
            "optimizer": optimizer_lr_details,
            "timing": {
                "train_started_at_unix": None,
                "train_finished_at_unix": None,
                "total_train_seconds": 0,
                "avg_epoch_seconds": 0,
            },
        }

    def _train_epoch(
        self, model, train_loader, optimizer, scheduler, scaler, amp_enabled
    ):
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = self._move_batch_to_device(batch)

            optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                outputs = model(**batch)
                loss = outputs["loss"]

            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config["training"]["max_grad_norm"]
            )

            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not amp_enabled or scaler.get_scale() >= prev_scale:
                scheduler.step()

        return total_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, amp_enabled):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                batch = self._move_batch_to_device(batch)
                labels = batch["labels"]

                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    outputs = model(**batch)
                    loss = outputs["loss"]

                total_loss += loss.item()

                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(all_labels, all_predictions)
        f1_weighted = f1_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        f1_macro = f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )

        return total_loss / len(val_loader), {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
        }

    def train(
        self,
        model,
        train_dataset,
        val_dataset,
        label2id: dict,
        id2label: dict,
        char_vocab: Optional[dict] = None,
    ):
        """
        Main training loop.

        Args:
            model: PyTorch model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            label2id: Label to ID mapping
            id2label: ID to label mapping
            char_vocab: Character vocabulary (optional)
        """
        self.utils.log("Trainer", LogType.INFO, "Starting training...")

        train_loader, val_loader = self._create_dataloaders(train_dataset, val_dataset)
        optimizer, optimizer_lr_details = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, train_loader)
        scaler = GradScaler(enabled=self.device.type == "cuda")

        amp_enabled = self.device.type == "cuda"
        monitor, patience = self._resolve_early_stopping_config()

        training_results = self._init_training_results(optimizer_lr_details)
        best_score = float("inf") if monitor == "val_loss" else float("-inf")
        best_epoch = -1
        patience_counter = 0

        train_start_time = time.time()
        training_results["timing"]["train_started_at_unix"] = train_start_time

        num_epochs = int(self.config["training"]["num_epochs"])

        for epoch in range(num_epochs):
            epoch_start = time.time()

            train_loss = self._train_epoch(
                model, train_loader, optimizer, scheduler, scaler, amp_enabled
            )
            val_loss, val_metrics = self._validate_epoch(model, val_loader, amp_enabled)

            epoch_duration = time.time() - epoch_start
            training_results["epoch_train_seconds"].append(epoch_duration)
            training_results["epochs"].append(epoch)
            training_results["train_loss"].append(float(train_loss))
            training_results["val_loss"].append(float(val_loss))
            training_results["metrics"].append(val_metrics)

            # Determine monitor value
            if monitor == "val_loss":
                current_score = val_loss
                is_improvement = current_score < best_score
            elif monitor == "val_f1_weighted":
                current_score = val_metrics["f1_weighted"]
                is_improvement = current_score > best_score
            else:  # val_f1_macro
                current_score = val_metrics["f1_macro"]
                is_improvement = current_score > best_score

            if is_improvement:
                best_score = current_score
                best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                self.checkpoint_manager.save_best_checkpoint(
                    model,
                    optimizer,
                    monitor_value=best_score,
                    monitor_name=monitor,
                    label2id=label2id,
                    id2label=id2label,
                    char_vocab=char_vocab or {},
                )
            else:
                patience_counter += 1

            self.utils.log(
                "Trainer",
                LogType.INFO,
                f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"{monitor}={current_score:.4f} | patience={patience_counter}/{patience}",
            )

            if patience_counter >= patience:
                self.utils.log(
                    "Trainer",
                    LogType.INFO,
                    f"Early stopping triggered at epoch {epoch}",
                )
                break

        train_end_time = time.time()
        training_results["timing"]["train_finished_at_unix"] = train_end_time
        training_results["timing"]["total_train_seconds"] = (
            train_end_time - train_start_time
        )
        num_completed_epochs = len(training_results["epochs"])
        training_results["timing"]["avg_epoch_seconds"] = (
            training_results["timing"]["total_train_seconds"] / num_completed_epochs
            if num_completed_epochs > 0
            else 0
        )

        # Save last checkpoint
        self.checkpoint_manager.save_last_checkpoint(
            model,
            optimizer,
            label2id=label2id,
            id2label=id2label,
            char_vocab=char_vocab or {},
        )

        # Save training results
        self.checkpoint_manager.save_training_results(
            training_results, label2id, id2label
        )

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Training completed. Best epoch: {best_epoch}, {monitor}={best_score:.4f}",
        )

        return training_results

    def load_best_model(self, model_builder):
        """
        Load best model from checkpoint.

        Args:
            model_builder: ModelBuilder instance to build model architecture

        Returns:
            Loaded model in eval mode
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            self.utils.log(
                "Trainer",
                LogType.WARNING,
                f"Best model not found at {best_path}, attempting to use last model",
            )
            best_path = self.checkpoint_dir / "last_model.pt"

        checkpoint = self.checkpoint_manager.load_checkpoint(best_path)
        label2id = checkpoint.get("label2id", {})
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)
        char_vocab = checkpoint.get("char_vocab", {})

        # Rebuild model architecture
        model = model_builder.build_model(
            config_model=self.config["model"],
            num_labels=num_labels,
            bert_model=None,
            char_vocab_size=len(char_vocab),
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Best model loaded from {best_path.name}",
        )

        return model, id2label

    def evaluate(self, model, test_dataset, id2label: dict):
        """
        Evaluate model on test dataset.

        Args:
            model: PyTorch model
            test_dataset: Test dataset
            id2label: ID to label mapping

        Returns:
            Evaluation results dict
        """
        self.utils.log("Trainer", LogType.INFO, "Evaluating...")

        model.eval()
        batch_size = self.config["training"]["batch_size"]
        amp_enabled = self.device.type == "cuda"
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=amp_enabled,
        )

        # Run inference
        all_labels, all_predictions, all_tokens = self.evaluator.run_inference(
            model, test_loader, amp_enabled
        )

        # Compute metrics
        eval_results, class_report, ordered_labels = self.evaluator.compute_metrics(
            all_labels, all_predictions, id2label
        )

        # Build and save prediction artifacts
        prediction_rows = self.evaluator.build_prediction_rows(
            all_labels, all_predictions, all_tokens, id2label
        )
        self.evaluator.save_prediction_artifacts(
            prediction_rows, all_labels, all_predictions, ordered_labels
        )

        # Save evaluation results
        self.checkpoint_manager.save_evaluation_results(eval_results)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Eval Results:\n{class_report}",
        )

        return eval_results
