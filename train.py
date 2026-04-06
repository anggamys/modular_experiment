import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data import Data
from hugging_face import HuggingFace
from model import IndoBERTForTokenClassification
from type import LogType
from utils import Utils


class TokenDataset(Dataset):
    """Dataset class for token classification tasks."""

    def __init__(
        self,
        tokens: list,
        labels: list,
        label2id: dict,
        tokenizer,
        max_length: int = 128,
    ):
        self.tokens = tokens
        self.labels = labels
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = str(self.tokens[idx])
        label = self.labels[idx]
        label_id = self.label2id[label]

        # Tokenize the token
        encoding = self.tokenizer(
            token,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros(self.max_length)
            ).squeeze(),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


class Trainer:
    """Trainer class for model training and evaluation."""

    def __init__(self, config_path: str):
        self.utils = Utils()
        self.data = Data()
        self.hugging_face = HuggingFace()

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.config["training"]["device"] == "cuda"
            else "cpu"
        )
        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Using device: {self.device}",
        )

        # Set seed
        self._set_seed(self.config["experiment"]["seed"])

        # Create output directories
        self.checkpoint_dir = Path(self.config["output"]["model_save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Checkpoint directory: {self.checkpoint_dir}",
        )

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def prepare_data(
        self, csv_path: str
    ) -> Tuple[TokenDataset, TokenDataset, TokenDataset, dict, dict]:
        """
        Prepare training, validation, and test datasets.

        Args:
            csv_path: Path to the CSV file containing tokens and labels

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset, label2id, id2label)
        """
        self.utils.log("Trainer", LogType.INFO, f"Loading data from {csv_path}")

        # Load data
        df = self.data.load_data(csv_path)
        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Data shape: {df.shape}",
        )

        # Create label mappings
        labels = df["final_pos_tag"].unique().tolist()
        label2id = self.data.label2id(labels)
        id2label = self.data.id2label(labels)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Number of labels: {len(label2id)} | Label to ID mapping: {label2id}",
        )

        # Download model and get tokenizer
        model_path = self.hugging_face.huggingface_download(
            self.config["model"]["model_name"]
        )
        tokenizer = self.hugging_face.tokenizer(model_path)

        # Split data: train-val-test
        test_size = self.config["data"]["test_size"]
        val_size = self.config["data"]["validation_size"]
        random_state = self.config["data"]["random_state"]

        # First split: separate test set
        train_val_tokens, test_tokens, train_val_labels, test_labels = train_test_split(
            df["token"],
            df["final_pos_tag"],
            test_size=test_size,
            random_state=random_state,
        )

        # Second split: separate validation from training
        train_tokens, val_tokens, train_labels, val_labels = train_test_split(
            train_val_tokens,
            train_val_labels,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
        )

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Train set size: {len(train_tokens)} | Val set size: {len(val_tokens)} | Test set size: {len(test_tokens)}",
        )

        # Create datasets
        train_dataset = TokenDataset(
            train_tokens.tolist(),
            train_labels.tolist(),
            label2id,
            tokenizer,
            self.config["data"]["max_length"],
        )
        val_dataset = TokenDataset(
            val_tokens.tolist(),
            val_labels.tolist(),
            label2id,
            tokenizer,
            self.config["data"]["max_length"],
        )
        test_dataset = TokenDataset(
            test_tokens.tolist(),
            test_labels.tolist(),
            label2id,
            tokenizer,
            self.config["data"]["max_length"],
        )

        return train_dataset, val_dataset, test_dataset, label2id, id2label

    def train(
        self,
        train_dataset: TokenDataset,
        val_dataset: TokenDataset,
        label2id: dict,
        id2label: dict,
        model_path: str,
    ):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            label2id: Label to ID mapping
            id2label: ID to label mapping
            model_path: Path to pretrained model
        """
        self.utils.log("Trainer", LogType.INFO, "Starting training...")

        # Update config with actual number of labels
        self.config["model"]["num_labels"] = len(label2id)

        # Load pretrained model
        bert_model = self.hugging_face.model(model_path)

        # Create model
        model = IndoBERTForTokenClassification(
            bert_model,
            num_labels=self.config["model"]["num_labels"],
            hidden_size=self.config["model"]["hidden_size"],
        )

        # Freeze BERT if configured
        if self.config["model"]["freeze_bert"]:
            model.freeze_bert_encoder()

        model.to(self.device)

        # Create data loaders
        try:
            batch_size = int(self.config["training"]["batch_size"])
            learning_rate = float(self.config["training"]["learning_rate"])
            weight_decay = float(self.config["training"]["weight_decay"])
            num_epochs = int(self.config["training"]["num_epochs"])
            warmup_steps = int(self.config["training"]["warmup_steps"])
        except (TypeError, ValueError) as error:
            self.utils.log(
                "Trainer",
                LogType.ERROR,
                f"Invalid numeric value in training config: {error}",
            )
            raise SystemExit(1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        training_results = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        for epoch in range(num_epochs):
            self.utils.log(
                "Trainer",
                LogType.INFO,
                f"Epoch {epoch + 1}/{num_epochs}",
            )

            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler)

            # Validation phase
            val_loss, val_metrics = self._validate_epoch(model, val_loader, id2label)

            training_results["epochs"].append(epoch + 1)
            training_results["train_loss"].append(train_loss)
            training_results["val_loss"].append(val_loss)
            training_results["val_accuracy"].append(val_metrics["accuracy"])
            training_results["val_f1"].append(val_metrics["f1"])

            self.utils.log(
                "Trainer",
                LogType.INFO,
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}",
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                        "label2id": label2id,
                        "id2label": id2label,
                    },
                    checkpoint_path,
                )

                self.utils.log(
                    "Trainer",
                    LogType.INFO,
                    f"Model saved to {checkpoint_path}",
                )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                self.utils.log(
                    "Trainer",
                    LogType.WARNING,
                    f"Early stopping at epoch {epoch + 1}",
                )
                break

        # Save training results
        results_path = self.checkpoint_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=2)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"Training results saved to {results_path}",
        )

        return model, label2id, id2label, training_results

    def _train_epoch(self, model, train_loader, optimizer, scheduler):
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs["loss"]
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config["training"]["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()

        return total_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, id2label):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

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

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
        }

        return total_loss / len(val_loader), metrics

    def evaluate(self, model, test_dataset, id2label):
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            test_dataset: Test dataset
            id2label: ID to label mapping

        Returns:
            dict: Evaluation metrics
        """
        self.utils.log("Trainer", LogType.INFO, "Evaluating on test set...")

        model.eval()
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
        )

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

        # Classification report
        class_report = classification_report(
            all_labels,
            all_predictions,
            target_names=[id2label[str(i)] for i in range(len(id2label))],
            zero_division=0,
        )

        evaluation_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": class_report,
        }

        # Save evaluation results
        eval_path = self.checkpoint_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            eval_dict = {
                k: v
                for k, v in evaluation_results.items()
                if k != "classification_report"
            }
            json.dump(eval_dict, f, indent=2)

        self.utils.log(
            "Trainer",
            LogType.INFO,
            f"\nTest Metrics:\n{class_report}",
        )

        return evaluation_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train IndoBERT for POS tagging")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset CSV file",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = Trainer(args.config)

    # Prepare data
    train_dataset, val_dataset, test_dataset, label2id, id2label = trainer.prepare_data(
        args.dataset
    )

    # Get model path
    hugging_face = HuggingFace()
    model_path = hugging_face.huggingface_download(
        trainer.config["model"]["model_name"]
    )

    # Train
    model, label2id, id2label, training_results = trainer.train(
        train_dataset, val_dataset, label2id, id2label, model_path
    )

    # Evaluate
    eval_results = trainer.evaluate(model, test_dataset, id2label)

    trainer.utils.log(
        "Main",
        LogType.INFO,
        f"Training completed! Results saved to {trainer.checkpoint_dir}",
    )


if __name__ == "__main__":
    main()
