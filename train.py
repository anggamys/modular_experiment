import json
import time
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

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

        # Always keep an experiment-local log in checkpoint dir.
        # If --log_file is enabled in main, logs will be mirrored to both files.
        self.utils.log2file(log_dir=str(self.checkpoint_dir), filename="run.log")

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Device: {self.device} | Path: {self.checkpoint_dir}",
        )
        self._apply_epoch_policy()
        self._apply_learning_rate_policy()

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
        # By default, keep user-defined upper bound and only enforce a minimum.
        # Set enforce_epoch_policy_cap=true to also enforce architecture max cap.
        enforce_cap = bool(training_cfg.get("enforce_epoch_policy_cap", False))
        if enforce_cap:
            adjusted_epochs = max(min_epoch, min(current_epochs, max_epoch))
        else:
            adjusted_epochs = max(min_epoch, current_epochs)

        if adjusted_epochs != current_epochs:
            self.utils.log(
                "Trainer",
                LogType.WARNING,
                f"num_epochs={current_epochs} adjusted to {adjusted_epochs} for architecture={architecture}",
            )

        self.config["training"]["num_epochs"] = adjusted_epochs

    def _apply_learning_rate_policy(self):
        training_cfg = self.config.get("training", {})
        if not training_cfg.get("enforce_learning_rate_policy", True):
            return

        architecture = str(self.config.get("model", {}).get("architecture", "")).lower()
        model_name = str(self.config.get("model", {}).get("model_name", "")).lower()

        if architecture.startswith("char_"):
            if architecture == "char_cnn":
                recommended_lr = 1e-3
            elif architecture == "char_bilstm":
                recommended_lr = 7.5e-4
            else:
                recommended_lr = 5e-4
        elif architecture == "hybrid_bert_charcnn":
            recommended_lr = 3e-5 if "distilbert" in model_name else 2e-5
        else:
            recommended_lr = 3e-5 if "distilbert" in model_name else 2e-5

        current_lr = float(training_cfg.get("learning_rate", recommended_lr))
        if current_lr != recommended_lr:
            self.utils.log(
                "Trainer",
                LogType.WARNING,
                f"learning_rate={current_lr} adjusted to {recommended_lr} for architecture={architecture}",
            )
        self.config["training"]["learning_rate"] = recommended_lr

        self.config["training"].setdefault("bert_learning_rate", 2e-5)
        self.config["training"].setdefault("head_learning_rate", 1e-4)
        self.config["training"].setdefault("char_learning_rate", 1e-3)

    def _build_optimizer(self, model):
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

                param_groups = []
                if bert_params:
                    param_groups.append(
                        {
                            "params": bert_params,
                            "lr": bert_lr,
                            "weight_decay": weight_decay,
                        }
                    )
                if char_params:
                    param_groups.append(
                        {
                            "params": char_params,
                            "lr": char_lr,
                            "weight_decay": weight_decay,
                        }
                    )
                if head_params:
                    param_groups.append(
                        {
                            "params": head_params,
                            "lr": head_lr,
                            "weight_decay": weight_decay,
                        }
                    )

                optimizer = AdamW(param_groups)
                self.utils.log(
                    "Trainer",
                    LogType.INFO,
                    f"Optimizer split LR | bert={bert_lr} char={char_lr} head={head_lr}",
                )
                return optimizer, {
                    "strategy": "split-hybrid",
                    "effective_param_group_lrs": [
                        {"name": "bert", "lr": bert_lr},
                        {"name": "char", "lr": char_lr},
                        {"name": "head", "lr": head_lr},
                    ],
                }

            head_params = [
                p for p in model.parameters() if p.requires_grad and id(p) not in bert_param_ids
            ]
            param_groups = []
            if bert_params:
                param_groups.append(
                    {"params": bert_params, "lr": bert_lr, "weight_decay": weight_decay}
                )
            if head_params:
                param_groups.append(
                    {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
                )

            optimizer = AdamW(param_groups)
            self.utils.log(
                "Trainer",
                LogType.INFO,
                f"Optimizer split LR | bert={bert_lr} head={head_lr}",
            )
            return optimizer, {
                "strategy": "split-bert-head",
                "effective_param_group_lrs": [
                    {"name": "bert", "lr": bert_lr},
                    {"name": "head", "lr": head_lr},
                ],
            }

        optimizer = AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )
        self.utils.log("Trainer", LogType.INFO, f"Optimizer LR (global): {base_lr}")
        return optimizer, {
            "strategy": "global",
            "effective_param_group_lrs": [{"name": "global", "lr": base_lr}],
        }

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

    def _create_dataloaders(self, train_dataset, val_dataset):
        batch_size = int(self.config["training"]["batch_size"])
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
        return train_loader, val_loader

    def _build_scheduler(self, optimizer, train_loader):
        total_steps = len(train_loader) * self.config["training"]["num_epochs"]
        warmup_steps = float(self.config["training"]["warmup_steps"])

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _resolve_early_stopping_config(self):
        early_stopping_monitor = self.config["training"].get(
            "early_stopping_monitor", "val_loss"
        )
        early_stopping_mode = self.config["training"].get("early_stopping_mode", "min")
        patience = int(self.config["training"].get("early_stopping_patience", 2))

        if early_stopping_mode not in {"min", "max"}:
            self.utils.log(
                "Trainer",
                LogType.WARNING,
                f"Invalid early_stopping_mode={early_stopping_mode}, fallback to 'min'",
            )
            early_stopping_mode = "min"

        self.utils.log(
            "Trainer",
            LogType.INFO,
            (
                f"Training setup | epochs={self.config['training']['num_epochs']} "
                f"| early_stopping_monitor={early_stopping_monitor} "
                f"| mode={early_stopping_mode} | patience={patience}"
            ),
        )
        return early_stopping_monitor, early_stopping_mode, patience

    def _init_training_results(self, optimizer_lr_details):
        return {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_weighted": [],
            "val_f1_macro": [],
            "epoch_train_seconds": [],
            "optimizer": optimizer_lr_details,
        }

    def _save_best_checkpoint(
        self,
        epoch,
        model,
        optimizer,
        early_stopping_monitor,
        early_stopping_mode,
        current_score,
        val_loss,
        label2id,
        id2label,
        char_vocab,
    ):
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "monitor": early_stopping_monitor,
            "monitor_mode": early_stopping_mode,
            "monitor_value": current_score,
            "val_loss": val_loss,
            "label2id": label2id,
            "id2label": id2label,
            "char_vocab": char_vocab,
        }
        best_model_path = self.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint_payload, best_model_path)
        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Updated best: {best_model_path.name} (epoch {epoch})",
        )
        return best_model_path.name

    def _save_last_checkpoint(self, epoch, model, optimizer, label2id, id2label, char_vocab):
        last_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
            "char_vocab": char_vocab,
        }
        last_model_path = self.checkpoint_dir / "last_model.pt"
        torch.save(last_payload, last_model_path)
        self.utils.log("Trainer", LogType.INFO, f"Saved last: {last_model_path.name}")

    def _save_training_artifacts(
        self,
        results,
        train_started_at,
        early_stopping_monitor,
        early_stopping_mode,
        best_epoch,
        best_checkpoint_name,
        best_score,
        last_epoch,
    ):
        train_finished_at = time.time()
        total_train_seconds = train_finished_at - train_started_at
        trained_epochs = len(results["epochs"])
        avg_epoch_seconds = (
            total_train_seconds / trained_epochs if trained_epochs > 0 else 0.0
        )

        results["timing"] = {
            "train_started_at_unix": train_started_at,
            "train_finished_at_unix": train_finished_at,
            "total_train_seconds": total_train_seconds,
            "avg_epoch_seconds": avg_epoch_seconds,
        }

        results_path = self.checkpoint_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if best_epoch is not None:
            best_info = {
                "best_epoch": best_epoch,
                "best_checkpoint": best_checkpoint_name,
                "best_alias": "best_model.pt",
                "last_epoch": last_epoch,
                "last_checkpoint": "last_model.pt",
                "monitor": early_stopping_monitor,
                "monitor_mode": early_stopping_mode,
                "best_score": best_score,
            }
            best_info_path = self.checkpoint_dir / "best_model_info.json"
            with open(best_info_path, "w") as f:
                json.dump(best_info, f, indent=2)
            self.utils.log("Trainer", LogType.INFO, f"Best model info: {best_info_path}")

        self.utils.log("Trainer", LogType.INFO, f"Results: {results_path}")

    def train(
        self,
        model,
        train_dataset,
        val_dataset,
        label2id,
        id2label,
        class_weights=None,
        char_vocab=None,
    ):
        self.utils.log("Trainer", LogType.INFO, "Starting training...")
        train_started_at = time.time()

        model.to(self.device)

        if class_weights is not None:
            weight_tensor = (
                class_weights.to(self.device)
                if torch.is_tensor(class_weights)
                else torch.tensor(class_weights, dtype=torch.float, device=self.device)
            )
            model.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            self.utils.log("Trainer", LogType.INFO, "Class-weighted loss enabled")

        train_loader, val_loader = self._create_dataloaders(train_dataset, val_dataset)

        optimizer, optimizer_lr_details = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, train_loader)
        amp_enabled = self.device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)
        early_stopping_monitor, early_stopping_mode, patience = (
            self._resolve_early_stopping_config()
        )

        best_score = float("inf") if early_stopping_mode == "min" else float("-inf")
        patience_counter = 0
        best_epoch = None
        best_checkpoint_name = None

        results: dict[str, Any] = self._init_training_results(optimizer_lr_details)

        for epoch in range(self.config["training"]["num_epochs"]):
            epoch_started_at = time.time()
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
            results["epoch_train_seconds"].append(time.time() - epoch_started_at)

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

                best_checkpoint_name = self._save_best_checkpoint(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    early_stopping_monitor=early_stopping_monitor,
                    early_stopping_mode=early_stopping_mode,
                    current_score=current_score,
                    val_loss=val_loss,
                    label2id=label2id,
                    id2label=id2label,
                    char_vocab=char_vocab,
                )
                best_epoch = epoch + 1
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.utils.log(
                    "Trainer",
                    LogType.WARNING,
                    f"Early stopping at epoch {epoch + 1}",
                )
                break

        last_epoch = len(results["epochs"])
        if last_epoch > 0:
            self._save_last_checkpoint(
                epoch=last_epoch,
                model=model,
                optimizer=optimizer,
                label2id=label2id,
                id2label=id2label,
                char_vocab=char_vocab,
            )

        self._save_training_artifacts(
            results=results,
            train_started_at=train_started_at,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            best_epoch=best_epoch,
            best_checkpoint_name=best_checkpoint_name,
            best_score=best_score,
            last_epoch=last_epoch,
        )

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
        f1_macro = f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )

        return total_loss / len(val_loader), {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
        }

    def _run_eval_inference(self, model, test_loader, amp_enabled):
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

        return all_labels, all_predictions

    def _build_eval_metrics(self, all_labels, all_predictions, id2label):
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
        f1_macro = f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )

        ordered_labels = [
            id2label.get(i, id2label.get(str(i), str(i))) for i in range(len(id2label))
        ]
        label_ids = list(range(len(ordered_labels)))

        class_report = classification_report(
            all_labels,
            all_predictions,
            labels=label_ids,
            target_names=ordered_labels,
            zero_division=0,
        )

        per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=label_ids,
            average=None,
            zero_division=0,
        )

        per_precision_arr = np.asarray(per_precision, dtype=float)
        per_recall_arr = np.asarray(per_recall, dtype=float)
        per_f1_arr = np.asarray(per_f1, dtype=float)
        if per_support is None:
            per_support_arr = np.zeros(len(label_ids), dtype=int)
        else:
            per_support_arr = np.asarray(per_support, dtype=int)

        per_label_metrics = []
        for i, label_id in enumerate(label_ids):
            per_label_metrics.append(
                {
                    "id": label_id,
                    "label": ordered_labels[i],
                    "precision": float(per_precision_arr[i]),
                    "recall": float(per_recall_arr[i]),
                    "f1": float(per_f1_arr[i]),
                    "support": int(per_support_arr[i]),
                }
            )

        eval_results = {
            "summary": {
                "num_samples": len(all_labels),
                "num_labels": len(ordered_labels),
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1_weighted),
                "f1_macro": float(f1_macro),
            },
            "per_label": per_label_metrics,
        }

        return eval_results, class_report, ordered_labels

    def _build_prediction_rows(self, all_labels, all_predictions, id2label):
        prediction_rows = []
        for index, (true_id, pred_id) in enumerate(zip(all_labels, all_predictions)):
            true_label = id2label.get(true_id, id2label.get(str(true_id), str(true_id)))
            pred_label = id2label.get(pred_id, id2label.get(str(pred_id), str(pred_id)))
            prediction_rows.append(
                {
                    "index": index,
                    "ground_truth_id": int(true_id),
                    "ground_truth_label": true_label,
                    "predicted_id": int(pred_id),
                    "predicted_label": pred_label,
                    "is_correct": int(true_id == pred_id),
                }
            )
        return prediction_rows

    def _save_prediction_artifacts(
        self, all_labels, all_predictions, ordered_labels, prediction_rows
    ):
        predictions_csv_path = self.checkpoint_dir / "evaluation_predictions.csv"
        with open(predictions_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "ground_truth_id",
                    "ground_truth_label",
                    "predicted_id",
                    "predicted_label",
                    "is_correct",
                ],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)

        predictions_json_path = self.checkpoint_dir / "evaluation_predictions.json"
        self.utils.write_json(
            predictions_json_path,
            {
                "labels": ordered_labels,
                "y_true": [int(x) for x in all_labels],
                "y_pred": [int(x) for x in all_predictions],
                "rows": prediction_rows,
            },
        )

        return predictions_csv_path, predictions_json_path

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

        all_labels, all_predictions = self._run_eval_inference(
            model, test_loader, amp_enabled
        )
        eval_results, class_report, ordered_labels = self._build_eval_metrics(
            all_labels, all_predictions, id2label
        )
        prediction_rows = self._build_prediction_rows(
            all_labels, all_predictions, id2label
        )

        predictions_csv_path, predictions_json_path = self._save_prediction_artifacts(
            all_labels, all_predictions, ordered_labels, prediction_rows
        )

        eval_path = self.checkpoint_dir / "evaluation_results.json"
        self.utils.write_json(eval_path, eval_results)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Eval Results:\n{class_report}",
        )
        
        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Saved prediction artifacts: {predictions_csv_path.name}, {predictions_json_path.name}",
        )

        return eval_results
