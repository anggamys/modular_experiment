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
    def __init__(self, config_path: str):
        self.utils = Utils()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.config["training"]["device"] == "cuda"
            else "cpu"
        )

        self._set_seed(self.config["experiment"]["seed"])

        self.checkpoint_dir = Path(self.config["output"]["model_save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Device: {self.device} | Path: {self.checkpoint_dir}",
        )

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        results = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
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
            results["val_f1"].append(val_metrics["f1"])

            self.utils.log(
                "Trainer",
                LogType.INFO,
                f"Ep {epoch + 1} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                ckpt_path = self.checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
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
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                loss = outputs["loss"]

            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config["training"]["max_grad_norm"]
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        return total_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, amp_enabled):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )
                    loss = outputs["loss"]

                total_loss += loss.item()

                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

        return total_loss / len(val_loader), {"accuracy": accuracy, "f1": f1}

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
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

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

        f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

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
            "f1": f1,
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
