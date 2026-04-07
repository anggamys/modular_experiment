import os
import torch
import pandas as pd
from torch.utils.data import Dataset
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
                "token_type_ids", torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
