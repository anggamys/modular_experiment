import os


import pandas as pd
from type import LogType
from utils import Utils


class Data:
    def __init__(self) -> None:
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
