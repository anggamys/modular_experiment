"""
Inference script untuk menggunakan trained model untuk prediksi.
"""

import argparse
from typing import Any, Dict, List

import torch

from hugging_face import HuggingFace
from model import IndoBERTForTokenClassification
from type import LogType
from utils import Utils


class Inference:
    """Inference wrapper untuk model POS tagging."""

    def __init__(
        self, checkpoint_path: str, model_name: str = "indolem/indobert-base-p1"
    ):
        self.utils = Utils()
        self.hugging_face = HuggingFace()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        self.utils.log(
            "Inference",
            LogType.INFO,
            f"Loading checkpoint from {checkpoint_path}",
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load label mappings
        self.label2id = checkpoint.get("label2id", {})
        self.id2label = {int(k): v for k, v in checkpoint.get("id2label", {}).items()}

        self.utils.log(
            "Inference",
            LogType.INFO,
            f"Labels: {self.id2label}",
        )

        # Load tokenizer
        model_path = self.hugging_face.huggingface_download(model_name)
        self.tokenizer = self.hugging_face.tokenizer(model_path)

        # Load model
        bert_model = self.hugging_face.model(model_path)
        self.model = IndoBERTForTokenClassification(
            bert_model,
            num_labels=len(self.id2label),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.utils.log(
            "Inference",
            LogType.INFO,
            "Model loaded successfully",
        )

    def predict(self, token: str) -> Dict[str, Any]:
        """
        Predict POS tag untuk token.

        Args:
            token: Input token

        Returns:
            dict: Prediction hasil dengan confidence scores
        """
        # Tokenize
        encoding = self.tokenizer(
            str(token),
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = torch.as_tensor(encoding["input_ids"], device=self.device)
        attention_mask = torch.as_tensor(encoding["attention_mask"], device=self.device)

        token_type_ids_raw = encoding.get("token_type_ids")
        if token_type_ids_raw is None:
            token_type_ids = torch.zeros_like(input_ids, device=self.device)
        else:
            token_type_ids = torch.as_tensor(token_type_ids_raw, device=self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        pred_label = self.id2label[int(pred_id)]
        confidence = probs[0, int(pred_id)].item()

        # Get top-3 predictions
        top_probs, top_ids = torch.topk(probs[0], k=min(3, len(self.id2label)))

        result = {
            "token": str(token),
            "predicted_label": pred_label,
            "confidence": float(confidence),
            "top_predictions": [
                {
                    "label": self.id2label[int(top_ids[i].item())],
                    "confidence": float(top_probs[i].item()),
                }
                for i in range(len(top_ids))
            ],
        }

        return result

    def predict_batch(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Batch prediction.

        Args:
            tokens: List of tokens

        Returns:
            list: List of predictions
        """
        results = []
        for token in tokens:
            result = self.predict(token)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Inference dengan trained POS tagging model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    parser.add_argument(
        "--token",
        type=str,
        help="Single token untuk prediksi",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        nargs="+",
        help="Multiple tokens untuk batch prediction",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="indolem/indobert-base-p1",
        help="Model name dari HuggingFace",
    )

    args = parser.parse_args()

    # Initialize inference
    inference = Inference(args.checkpoint, args.model_name)

    # Predict
    if args.token:
        result = inference.predict(args.token)
        print(f"\nPrediction for token: {args.token}")
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nTop Predictions:")
        for i, pred in enumerate(result["top_predictions"], 1):
            print(f"  {i}. {pred['label']}: {pred['confidence']:.4f}")

    elif args.tokens:
        results = inference.predict_batch(args.tokens)
        print(f"\nBatch Prediction Results ({len(args.tokens)} tokens):")
        for result in results:
            print(
                f"Token: {result['token']:15} → {result['predicted_label']:10} (conf: {result['confidence']:.4f})"
            )

    else:
        parser.error("Either --token or --tokens must be provided")


if __name__ == "__main__":
    main()
