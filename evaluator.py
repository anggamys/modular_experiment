import csv
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from type import LogType
from utils import Utils


class Evaluator:
    def __init__(self, checkpoint_dir: Path, utils: Utils, device: torch.device):
        self.checkpoint_dir = checkpoint_dir
        self.utils = utils
        self.device = device

    def run_inference(
        self, model, test_loader, amp_enabled: bool = False
    ) -> Tuple[list, list, list]:
        all_predictions = []
        all_labels = []
        all_tokens = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                # Extract tokens before moving to device (tokens are strings, not tensors)
                tokens = batch.get("token", [])
                if isinstance(tokens, list):
                    all_tokens.extend(tokens)
                else:
                    all_tokens.extend(tokens if isinstance(tokens, list) else [tokens])

                batch = self._move_batch_to_device(batch)
                labels = batch["labels"]

                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    eval_batch = {
                        k: v for k, v in batch.items() if k not in ["labels", "token"]
                    }

                    outputs = model(**eval_batch)

                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        return all_labels, all_predictions, all_tokens

    def compute_metrics(
        self, all_labels: list, all_predictions: list, id2label: dict
    ) -> Tuple[dict, str, list]:
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

        class_report = cast(
            str,
            classification_report(
                all_labels,
                all_predictions,
                labels=label_ids,
                target_names=ordered_labels,
                output_dict=False,
                zero_division=0,
            ),
        )

        per_precision, per_recall, per_f1, per_support = (
            precision_recall_fscore_support(
                all_labels,
                all_predictions,
                labels=label_ids,
                average=None,
                zero_division=0,
            )
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

    def build_prediction_rows(
        self, all_labels: list, all_predictions: list, all_tokens: list, id2label: dict
    ) -> list:
        prediction_rows = []
        for index, (true_id, pred_id, token) in enumerate(
            zip(all_labels, all_predictions, all_tokens)
        ):
            true_label = id2label.get(true_id, id2label.get(str(true_id), str(true_id)))
            pred_label = id2label.get(pred_id, id2label.get(str(pred_id), str(pred_id)))
            prediction_rows.append(
                {
                    "index": index,
                    "token": str(token) if token is not None else "",
                    "ground_truth_id": int(true_id),
                    "ground_truth_label": true_label,
                    "predicted_id": int(pred_id),
                    "predicted_label": pred_label,
                    "is_correct": int(true_id == pred_id),
                }
            )

        return prediction_rows

    def save_prediction_artifacts(
        self,
        prediction_rows: list,
        all_labels: list,
        all_predictions: list,
        ordered_labels: list,
    ) -> Tuple[Path, Path]:
        # CSV output
        predictions_csv_path = self.checkpoint_dir / "evaluation_predictions.csv"
        with open(predictions_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "token",
                    "ground_truth_id",
                    "ground_truth_label",
                    "predicted_id",
                    "predicted_label",
                    "is_correct",
                ],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)

        # JSON output
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

        self.utils.log(
            "Evaluator",
            LogType.INFO,
            f"Prediction artifacts saved: {predictions_csv_path.name}, {predictions_json_path.name}",
        )

        return predictions_csv_path, predictions_json_path

    def _move_batch_to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
