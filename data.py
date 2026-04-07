import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from type import LogType
from utils import Utils


class Data:
    def __init__(self):
        self.utils = Utils()

    def label2id(self, labels: list) -> dict:
        return {label: idx for idx, label in enumerate(labels)}

    def id2label(self, labels: list) -> dict:
        return {idx: label for idx, label in enumerate(labels)}

    def load_data(self, path: str) -> pd.DataFrame:
        if not path:
            self.utils.log("Data", LogType.ERROR, "Dataset file path is required.")
            exit(1)

        if not os.path.exists(path):
            self.utils.log("Data", LogType.ERROR, f"File not found: {path}")
            exit(1)

        if not path.endswith(".csv"):
            self.utils.log("Data", LogType.ERROR, f"Unsupported file format: {path}")
            exit(1)

        try:
            return pd.read_csv(path)
        except Exception as e:
            self.utils.log("Data", LogType.ERROR, f"Failed to load CSV: {e}")
            exit(1)


class DataPipeline:
    def __init__(self):
        self.data = Data()
        self.utils = Utils()

    def prepare_datasets(
        self,
        csv_path: str,
        tokenizer,
        test_size: float,
        validation_size: float,
        random_state: int,
        max_length: int,
    ):
        self.utils.log("DataPipeline", LogType.INFO, f"Loading: {csv_path}")
        df = self.data.load_data(csv_path)
        self.utils.log("DataPipeline", LogType.INFO, f"Shape: {df.shape}")

        required_columns = {"token", "label"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            self.utils.log(
                "DataPipeline",
                LogType.ERROR,
                f"Missing required columns: {sorted(missing_columns)}",
            )

            self.utils.log(
                "DataPipeline",
                LogType.INFO,
                f"Expected columns: {sorted(required_columns)}, Found columns: {sorted(df.columns)}",
            )

            exit(1)

        labels = df["label"].unique().tolist()
        label2id = self.data.label2id(labels)
        id2label = self.data.id2label(labels)

        self.utils.log(
            "DataPipeline",
            LogType.INFO,
            f"Labels: {len(label2id)}, Mapping: {label2id}",
        )

        train_val_tokens, test_tokens, train_val_labels, test_labels = train_test_split(
            df["token"],
            df["label"],
            test_size=test_size,
            random_state=random_state,
        )

        train_tokens, val_tokens, train_labels, val_labels = train_test_split(
            train_val_tokens,
            train_val_labels,
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
        )

        self.utils.log(
            "DataPipeline",
            LogType.INFO,
            f"Split: train={len(train_tokens)} val={len(val_tokens)} test={len(test_tokens)}",
        )

        return (
            TokenDataset(
                train_tokens.tolist(),
                train_labels.tolist(),
                label2id,
                tokenizer,
                max_length,
            ),
            TokenDataset(
                val_tokens.tolist(),
                val_labels.tolist(),
                label2id,
                tokenizer,
                max_length,
            ),
            TokenDataset(
                test_tokens.tolist(),
                test_labels.tolist(),
                label2id,
                tokenizer,
                max_length,
            ),
            label2id,
            id2label,
        )


class TokenDataset(Dataset):
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
                "token_type_ids", torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
