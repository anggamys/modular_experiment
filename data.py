import os
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
        architecture: str,
        label_column: str = "label",
        char_max_length: int = 32,
        min_samples_per_label: int = 0,
        rare_label_strategy: str = "keep",
        use_class_weight: bool = True,
    ):
        self.utils.log("DataPipeline", LogType.INFO, f"Loading: {csv_path}")
        df = self.data.load_data(csv_path)
        self.utils.log("DataPipeline", LogType.INFO, f"Shape: {df.shape}")

        required_columns = {"token", label_column}
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

        label_counts_before_filter = df[label_column].value_counts().to_dict()
        raw_label_counts = df[label_column].value_counts().to_dict()
        rare_labels = sorted(
            [
                label
                for label, count in raw_label_counts.items()
                if count < int(min_samples_per_label)
            ],
            key=lambda item: str(item),
        )

        strategy = str(rare_label_strategy).lower()
        if min_samples_per_label > 0 and rare_labels:
            if strategy == "error":
                self.utils.log(
                    "DataPipeline",
                    LogType.ERROR,
                    f"Found {len(rare_labels)} labels with < {min_samples_per_label} samples: {rare_labels}",
                )
                exit(1)

            if strategy == "drop":
                before = len(df)
                df = df[~df[label_column].isin(rare_labels)].reset_index(drop=True)
                self.utils.log(
                    "DataPipeline",
                    LogType.WARNING,
                    f"Dropped {len(rare_labels)} rare labels (< {min_samples_per_label}) | rows: {before} -> {len(df)}",
                )
                if df.empty:
                    self.utils.log(
                        "DataPipeline",
                        LogType.ERROR,
                        "All rows removed after rare label filtering.",
                    )
                    exit(1)

            else:
                self.utils.log(
                    "DataPipeline",
                    LogType.WARNING,
                    f"Keeping rare labels (< {min_samples_per_label}) with strategy='{strategy}'.",
                )

        labels = df[label_column].unique().tolist()
        label2id = self.data.label2id(labels)
        id2label = self.data.id2label(labels)

        self.utils.log(
            "DataPipeline",
            LogType.INFO,
            f"Labels: {len(label2id)}, Mapping: {label2id}",
        )

        try:
            train_val_tokens, test_tokens, train_val_labels, test_labels = (
                train_test_split(
                    df["token"],
                    df[label_column],
                    test_size=test_size,
                    random_state=random_state,
                    stratify=df[label_column],
                )
            )

            train_tokens, val_tokens, train_labels, val_labels = train_test_split(
                train_val_tokens,
                train_val_labels,
                test_size=validation_size / (1 - test_size),
                random_state=random_state,
                stratify=train_val_labels,
            )
        except ValueError as error:
            self.utils.log(
                "DataPipeline",
                LogType.WARNING,
                f"Stratified split failed, fallback to random split: {error}",
            )
            train_val_tokens, test_tokens, train_val_labels, test_labels = (
                train_test_split(
                    df["token"],
                    df[label_column],
                    test_size=test_size,
                    random_state=random_state,
                )
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

        train_tokens = train_tokens.tolist()
        val_tokens = val_tokens.tolist()
        test_tokens = test_tokens.tolist()

        train_labels = train_labels.tolist()
        val_labels = val_labels.tolist()
        test_labels = test_labels.tolist()

        char_vocab = None
        if architecture in {
            "char_cnn",
            "char_bilstm",
            "char_cnn_bilstm",
            "hybrid_bert_charcnn",
        }:
            char_vocab = CharVocab.build(train_tokens)

        if architecture in {
            "bert_linear",
            "bert_mlp",
            "bert_gru",
            "bert_cnn",
        }:
            if tokenizer is None:
                self.utils.log(
                    "DataPipeline",
                    LogType.ERROR,
                    "Tokenizer is required for transformer-based architectures.",
                )
                exit(1)

            train_dataset = TokenDataset(
                train_tokens,
                train_labels,
                label2id,
                tokenizer,
                max_length,
            )
            val_dataset = TokenDataset(
                val_tokens,
                val_labels,
                label2id,
                tokenizer,
                max_length,
            )
            test_dataset = TokenDataset(
                test_tokens,
                test_labels,
                label2id,
                tokenizer,
                max_length,
            )

        elif architecture in {"char_cnn", "char_bilstm", "char_cnn_bilstm"}:
            if char_vocab is None:
                self.utils.log(
                    "DataPipeline",
                    LogType.ERROR,
                    "char_vocab is missing for char-level architecture.",
                )
                exit(1)

            train_dataset = CharDataset(
                train_tokens,
                train_labels,
                label2id,
                char_vocab,
                char_max_length,
            )
            val_dataset = CharDataset(
                val_tokens,
                val_labels,
                label2id,
                char_vocab,
                char_max_length,
            )
            test_dataset = CharDataset(
                test_tokens,
                test_labels,
                label2id,
                char_vocab,
                char_max_length,
            )

        elif architecture == "hybrid_bert_charcnn":
            if tokenizer is None:
                self.utils.log(
                    "DataPipeline",
                    LogType.ERROR,
                    "Tokenizer is required for hybrid architecture.",
                )
                exit(1)

            if char_vocab is None:
                self.utils.log(
                    "DataPipeline",
                    LogType.ERROR,
                    "char_vocab is missing for hybrid architecture.",
                )
                exit(1)

            train_dataset = HybridDataset(
                train_tokens,
                train_labels,
                label2id,
                tokenizer,
                max_length,
                char_vocab,
                char_max_length,
            )
            val_dataset = HybridDataset(
                val_tokens,
                val_labels,
                label2id,
                tokenizer,
                max_length,
                char_vocab,
                char_max_length,
            )
            test_dataset = HybridDataset(
                test_tokens,
                test_labels,
                label2id,
                tokenizer,
                max_length,
                char_vocab,
                char_max_length,
            )

        else:
            self.utils.log(
                "DataPipeline",
                LogType.ERROR,
                f"Unsupported architecture: {architecture}",
            )

            exit(1)

        metadata = {
            "char_vocab_size": len(char_vocab) if char_vocab is not None else None,
            "dropped_labels": rare_labels,
            "label_counts": df[label_column].value_counts().to_dict(),
            "class_weights": None,
        }

        if use_class_weight:
            train_counts = Counter(train_labels)
            total_train = len(train_labels)
            num_classes = len(label2id)
            class_weights = []
            for class_id in range(num_classes):
                label = id2label[class_id]
                count = train_counts.get(label, 0)
                weight = (
                    float(total_train) / float(num_classes * count)
                    if count > 0
                    else 0.0
                )
                class_weights.append(weight)

            metadata["class_weights"] = torch.tensor(class_weights, dtype=torch.float)
            metadata["class_weights_list"] = class_weights

        metadata["label_counts_before_filter"] = label_counts_before_filter
        metadata["train_label_counts"] = dict(Counter(train_labels))
        metadata["val_label_counts"] = dict(Counter(val_labels))
        metadata["test_label_counts"] = dict(Counter(test_labels))

        return train_dataset, val_dataset, test_dataset, label2id, id2label, metadata


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


class CharVocab:
    PAD = "<pad>"
    UNK = "<unk>"

    @classmethod
    def build(cls, tokens: list[str]) -> dict:
        chars = set()
        for token in tokens:
            chars.update(list(str(token)))

        vocab = {cls.PAD: 0, cls.UNK: 1}
        for index, char in enumerate(sorted(chars), start=2):
            vocab[char] = index

        return vocab

    @classmethod
    def encode(cls, token: str, vocab: dict, max_length: int):
        token = str(token)
        char_ids = [vocab.get(char, vocab[cls.UNK]) for char in token[:max_length]]
        mask = [1] * len(char_ids)

        pad_len = max_length - len(char_ids)
        if pad_len > 0:
            char_ids += [vocab[cls.PAD]] * pad_len
            mask += [0] * pad_len

        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
        )


class CharDataset(Dataset):
    def __init__(
        self,
        tokens: list,
        labels: list,
        label2id: dict,
        char_vocab: dict,
        char_max_length: int = 32,
    ):
        self.tokens = tokens
        self.labels = labels
        self.label2id = label2id
        self.char_vocab = char_vocab
        self.char_max_length = char_max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = str(self.tokens[idx])
        label_id = self.label2id[self.labels[idx]]
        char_ids, char_mask = CharVocab.encode(
            token, self.char_vocab, self.char_max_length
        )

        return {
            "char_ids": char_ids,
            "char_mask": char_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


class HybridDataset(Dataset):
    def __init__(
        self,
        tokens: list,
        labels: list,
        label2id: dict,
        tokenizer,
        max_length: int,
        char_vocab: dict,
        char_max_length: int = 32,
    ):
        self.tokens = tokens
        self.labels = labels
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.char_vocab = char_vocab
        self.char_max_length = char_max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = str(self.tokens[idx])
        label_id = self.label2id[self.labels[idx]]

        encoding = self.tokenizer(
            token,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        char_ids, char_mask = CharVocab.encode(
            token, self.char_vocab, self.char_max_length
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(),
            "char_ids": char_ids,
            "char_mask": char_mask,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
