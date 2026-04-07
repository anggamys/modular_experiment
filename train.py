import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from type import LogType
from utils import Utils


class Trainer:
    def __init__(self, config_path: str, exp_id: str | None = None):
        self.utils = Utils()

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

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

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Device: {self.device} | Path: {self.checkpoint_dir}",
        )
        self._apply_epoch_policy()

    def _resolve_experiment_config(self, raw_config: dict, exp_id: str | None):
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

    def _apply_epoch_policy(self):
        training_cfg = self.config.get("training", {})
        if not training_cfg.get("enforce_epoch_policy", True):
            return

        architecture = str(self.config.get("model", {}).get("architecture", "")).lower()
        policy = {
            "bert_linear": (3, 5),
            "bert_mlp": (3, 5),
            "bert_gru": (4, 6),
            "bert_cnn": (4, 6),
            "char_cnn": (8, 15),
            "char_bilstm": (8, 15),
            "char_cnn_bilstm": (10, 20),
            "hybrid_bert_charcnn": (4, 8),
        }
        min_epoch, max_epoch = policy.get(architecture, (3, 5))

        current_epochs = int(training_cfg.get("num_epochs", max_epoch))
        clamped_epochs = max(min_epoch, min(current_epochs, max_epoch))

        if clamped_epochs != current_epochs:
            self.utils.log(
                "Trainer",
                LogType.WARNING,
                f"num_epochs={current_epochs} adjusted to {clamped_epochs} for architecture={architecture}",
            )

        self.config["training"]["num_epochs"] = clamped_epochs

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _move_batch_to_device(self, batch: dict):
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def train(self, model, train_dataset, val_dataset, label2id, id2label):
        self.utils.log("Trainer", LogType.INFO, "Starting training...")

        model.to(self.device)

        batch_size = int(self.config["training"]["batch_size"])
        # Cap workers conservatively to avoid runtime warnings/freezes on limited environments.
        num_workers = min(2, max(0, len(train_dataset) // max(1, batch_size)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        optimizer = AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )

        total_steps = len(train_loader) * self.config["training"]["num_epochs"]
        warmup_steps = float(self.config["training"]["warmup_steps"])

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        amp_enabled = self.device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)

        early_stopping_monitor = self.config["training"].get(
            "early_stopping_monitor", "val_loss"
        )
        early_stopping_mode = self.config["training"].get(
            "early_stopping_mode", "min"
        )
        patience = int(self.config["training"].get("early_stopping_patience", 2))

        if early_stopping_mode not in {"min", "max"}:
            self.utils.log(
                "Trainer",
                LogType.WARNING,
                f"Invalid early_stopping_mode={early_stopping_mode}, fallback to 'min'",
            )
            early_stopping_mode = "min"

        best_score = float("inf") if early_stopping_mode == "min" else float("-inf")
        patience_counter = 0

        results = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_weighted": [],
            "val_f1_macro": [],
        }

        for epoch in range(self.config["training"]["num_epochs"]):
            train_loss = self._train_epoch(
                model, train_loader, optimizer, scheduler, scaler, amp_enabled
            )

            val_loss, val_metrics = self._validate_epoch(model, val_loader, amp_enabled)

            results["epochs"].append(epoch + 1)
            results["train_loss"].append(train_loss)
            results["val_loss"].append(val_loss)
            results["val_accuracy"].append(val_metrics["accuracy"])
            results["val_f1_weighted"].append(val_metrics["f1_weighted"])
            results["val_f1_macro"].append(val_metrics["f1_macro"])

            self.utils.log(
                "Trainer",
                LogType.INFO,
                f"Ep {epoch + 1} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1w: {val_metrics['f1_weighted']:.4f} | F1m: {val_metrics['f1_macro']:.4f}",
            )

            monitor_candidates = {
                "val_loss": val_loss,
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
            current_score = monitor_candidates.get(early_stopping_monitor, val_loss)

            improved = (
                current_score < best_score
                if early_stopping_mode == "min"
                else current_score > best_score
            )

            if improved:
                best_score = current_score
                patience_counter = 0

                ckpt_path = self.checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "monitor": early_stopping_monitor,
                        "monitor_mode": early_stopping_mode,
                        "monitor_value": current_score,
                        "val_loss": val_loss,
                        "label2id": label2id,
                        "id2label": id2label,
                    },
                    ckpt_path,
                )
                self.utils.log(
                    "Trainer",
                    LogType.INFO,
                    f"Saved: {ckpt_path.name}",
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.utils.log(
                    "Trainer",
                    LogType.WARNING,
                    f"Early stopping at epoch {epoch + 1}",
                )
                break

        results_path = self.checkpoint_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.utils.log("Trainer", LogType.INFO, f"Results: {results_path}")

        return model

    def _train_epoch(
        self, model, train_loader, optimizer, scheduler, scaler, amp_enabled
    ):
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

        accuracy = accuracy_score(all_labels, all_predictions)
        f1_weighted = f1_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

        return total_loss / len(val_loader), {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
        }

    def evaluate(self, model, test_dataset, id2label):
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

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                batch = self._move_batch_to_device(batch)
                labels = batch["labels"]

                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    eval_batch = {k: v for k, v in batch.items() if k != "labels"}
                    outputs = model(**eval_batch)

                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )

        recall = recall_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )

        f1_weighted = f1_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

        ordered_labels = [
            id2label.get(i, id2label.get(str(i), str(i))) for i in range(len(id2label))
        ]

        class_report = classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(ordered_labels))),
            target_names=ordered_labels,
            zero_division=0,
        )

        eval_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "classification_report": class_report,
        }

        eval_path = self.checkpoint_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Eval Results:\n{class_report}",
        )

        return eval_results
