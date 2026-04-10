import json
from pathlib import Path

import pandas as pd
import torch

from data import CharVocab
from hugging_face import HuggingFace
from model_builder import ModelBuilder
from train import Trainer
from type import LogType
from utils import Utils


def _select_checkpoint_path(checkpoint_dir: Path, checkpoint_name: str | None) -> Path:
    if checkpoint_name:
        candidate = Path(checkpoint_name)

        # If checkpoint_name is absolute path, use as-is
        if candidate.is_absolute():
            if not candidate.exists():
                raise FileNotFoundError(f"Checkpoint not found: {candidate}")
            return candidate

        # Otherwise, treat as relative to checkpoint_dir
        candidate = checkpoint_dir / checkpoint_name

        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")

        return candidate

    best = checkpoint_dir / "best_model.pt"
    if best.exists():
        return best

    raise FileNotFoundError(
        f"No best_model.pt found in {checkpoint_dir}. Please train first or pass --checkpoint_name."
    )


def _build_token_batch(token: str, tokenizer, max_length: int) -> dict:
    encoded = tokenizer(
        token,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded.get(
            "token_type_ids", torch.zeros((1, max_length), dtype=torch.long)
        ),
    }


def _build_char_batch(token: str, char_vocab: dict, char_max_length: int) -> dict:
    char_ids, char_mask = CharVocab.encode(token, char_vocab, char_max_length)

    return {
        "char_ids": char_ids.unsqueeze(0),
        "char_mask": char_mask.unsqueeze(0),
    }


def _prepare_model_and_assets(
    config_path: str, exp_id: str, checkpoint_name: str | None
):
    trainer = Trainer(config_path, exp_id=exp_id)
    config = trainer.config

    checkpoint_path = _select_checkpoint_path(trainer.checkpoint_dir, checkpoint_name)
    payload = torch.load(checkpoint_path, map_location=trainer.device)

    id2label = payload.get("id2label")
    if id2label is None:
        raise RuntimeError("Checkpoint does not include id2label.")

    if isinstance(id2label, dict):
        id2label = {
            int(key) if str(key).isdigit() else key: value
            for key, value in id2label.items()
        }

    architecture = config["model"]["architecture"]
    use_transformer = architecture in {
        "bert_linear",
        "bert_mlp",
        "bert_gru",
        "bert_cnn",
        "hybrid_bert_charcnn",
    }

    hugging_face = HuggingFace()
    tokenizer = None
    bert_model = None

    if use_transformer:
        model_path = hugging_face.huggingface_download(config["model"]["model_name"])
        tokenizer = hugging_face.tokenizer(model_path)
        bert_model = hugging_face.model(model_path)

    char_vocab = payload.get("char_vocab")

    if architecture.startswith("char_") or architecture == "hybrid_bert_charcnn":
        if not isinstance(char_vocab, dict) or len(char_vocab) == 0:
            raise RuntimeError(
                "char_vocab not found in checkpoint. Re-train to persist char vocab for char/hybrid annotation."
            )

    model_builder = ModelBuilder()
    model = model_builder.build_model(
        config_model=config["model"],
        num_labels=len(id2label),
        bert_model=bert_model,
        char_vocab_size=len(char_vocab) if isinstance(char_vocab, dict) else None,
    )

    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(trainer.device)
    model.eval()

    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "char_vocab": char_vocab,
        "id2label": id2label,
        "architecture": architecture,
        "checkpoint_path": checkpoint_path,
    }


def annotate_tokens(
    tokens: list[str],
    config_path: str,
    exp_id: str,
    checkpoint_name: str | None = None,
    confidence_threshold: float = 0.0,
    progress_callback=None,
    batch_size: int = 32,
):
    assets = _prepare_model_and_assets(config_path, exp_id, checkpoint_name)
    trainer = assets["trainer"]
    model = assets["model"]
    tokenizer = assets["tokenizer"]
    char_vocab = assets["char_vocab"]
    id2label = assets["id2label"]
    architecture = assets["architecture"]

    max_length = int(trainer.config["data"].get("max_length", 128))
    char_max_length = int(trainer.config["data"].get("char_max_length", 32))

    outputs = []
    total_tokens = len(tokens)
    processed_count = 0

    with torch.no_grad():
        # Process tokens in batches
        for batch_start in range(0, len(tokens), batch_size):
            batch_end = min(batch_start + batch_size, len(tokens))
            batch_tokens = tokens[batch_start:batch_end]
            batch_tokens = [str(t).strip() for t in batch_tokens if str(t).strip()]

            if not batch_tokens:
                continue

            # Prepare batch inputs
            batch_inputs = {}
            if architecture in {
                "bert_linear",
                "bert_mlp",
                "bert_gru",
                "bert_cnn",
            }:
                # Batch encode tokens
                encoded = tokenizer(
                    batch_tokens,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                batch_inputs["input_ids"] = encoded["input_ids"]
                batch_inputs["attention_mask"] = encoded["attention_mask"]
                batch_inputs["token_type_ids"] = encoded.get(
                    "token_type_ids",
                    torch.zeros((len(batch_tokens), max_length), dtype=torch.long),
                )

            elif architecture in {"char_cnn", "char_bilstm", "char_cnn_bilstm"}:
                char_ids_list = []
                char_mask_list = []
                for token in batch_tokens:
                    char_ids, char_mask = CharVocab.encode(
                        token, char_vocab, char_max_length
                    )
                    char_ids_list.append(char_ids)
                    char_mask_list.append(char_mask)
                batch_inputs["char_ids"] = torch.stack(char_ids_list)
                batch_inputs["char_mask"] = torch.stack(char_mask_list)

            elif architecture == "hybrid_bert_charcnn":
                # BERT inputs
                encoded = tokenizer(
                    batch_tokens,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                batch_inputs["input_ids"] = encoded["input_ids"]
                batch_inputs["attention_mask"] = encoded["attention_mask"]
                batch_inputs["token_type_ids"] = encoded.get(
                    "token_type_ids",
                    torch.zeros((len(batch_tokens), max_length), dtype=torch.long),
                )
                # Char inputs
                char_ids_list = []
                char_mask_list = []
                for token in batch_tokens:
                    char_ids, char_mask = CharVocab.encode(
                        token, char_vocab, char_max_length
                    )
                    char_ids_list.append(char_ids)
                    char_mask_list.append(char_mask)
                batch_inputs["char_ids"] = torch.stack(char_ids_list)
                batch_inputs["char_mask"] = torch.stack(char_mask_list)

            else:
                raise RuntimeError(
                    f"Unsupported architecture for annotation: {architecture}"
                )

            # Move batch to device
            batch_inputs = {
                key: value.to(trainer.device) if torch.is_tensor(value) else value
                for key, value in batch_inputs.items()
            }

            # Run inference
            with torch.autocast(
                device_type=trainer.device.type,
                enabled=trainer.device.type == "cuda",
            ):
                model_out = model(**batch_inputs)

            logits = model_out["logits"]
            probs = torch.softmax(logits, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = probs.max(dim=-1).values.cpu().numpy()

            # Process results for this batch
            for token, pred_idx, confidence in zip(
                batch_tokens, pred_indices, confidences
            ):
                processed_count += 1
                pred_idx = int(pred_idx)
                confidence = float(confidence)
                label = id2label.get(pred_idx, str(pred_idx))

                # Determine final label based on confidence threshold
                is_confident = confidence >= confidence_threshold
                final_label = label if is_confident else "UNID"

                outputs.append(
                    {
                        "token": token,
                        "predicted_label": label,
                        "predicted_label_id": pred_idx,
                        "confidence": confidence,
                        "passes_threshold": is_confident,
                        "final_label": final_label,
                    }
                )

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        current=processed_count,
                        total=total_tokens,
                        token=token,
                        label=final_label,
                        confidence=confidence,
                    )

            # Clear GPU memory between batches
            if trainer.device.type == "cuda":
                torch.cuda.empty_cache()

    return outputs, assets["checkpoint_path"]


def main():
    utils = Utils()
    args = utils.argument_parser(
        description="Annotate token(s) using trained POS model",
        arguments=[
            {
                "name": "--config",
                "help": "Path to config YAML",
                "type": str,
                "default": "config.yml",
                "required": False,
            },
            {
                "name": "--exp_id",
                "help": "Experiment ID (EXP-01 ... EXP-11)",
                "type": str,
                "required": True,
            },
            {
                "name": "--tokens",
                "help": "Input token(s), separated by spaces",
                "nargs": "+",
                "required": False,
            },
            {
                "name": "--input_csv",
                "help": "Path to CSV file containing token list",
                "type": str,
                "required": False,
            },
            {
                "name": "--token_column",
                "help": "Token column name in CSV input",
                "type": str,
                "default": "token",
                "required": False,
            },
            {
                "name": "--checkpoint_name",
                "help": "Checkpoint filename inside experiment checkpoint dir (default: best_model.pt)",
                "type": str,
                "required": False,
            },
            {
                "name": "--confidence_threshold",
                "help": "Confidence threshold (0.0-1.0) for accepting predictions. Below this = UNID (default: 0.0)",
                "type": float,
                "default": 0.0,
                "required": False,
            },
            {
                "name": "--output",
                "help": "Optional JSON output file path",
                "type": str,
                "required": False,
            },
        ],
    )

    # Validate confidence threshold
    confidence_threshold = float(args.confidence_threshold)
    if not (0.0 <= confidence_threshold <= 1.0):
        utils.log(
            "Annotate",
            LogType.ERROR,
            f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}",
        )
        return

    has_tokens_arg = bool(args.tokens and len(args.tokens) > 0)
    has_csv_arg = bool(args.input_csv and str(args.input_csv).strip())

    if not has_tokens_arg and not has_csv_arg:
        utils.log(
            "Annotate",
            LogType.ERROR,
            "Provide either --tokens or --input_csv.",
        )

        return

    if has_tokens_arg and has_csv_arg:
        utils.log(
            "Annotate",
            LogType.ERROR,
            "Use only one input source: --tokens or --input_csv.",
        )

        return

    source_rows = None
    if has_csv_arg:
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            utils.log("Annotate", LogType.ERROR, f"CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        token_column = args.token_column

        if token_column not in df.columns:
            utils.log(
                "Annotate",
                LogType.ERROR,
                f"Column '{token_column}' not found in CSV. Available columns: {list(df.columns)}",
            )

            return

        source_rows = df.copy()
        tokens = df[token_column].astype(str).map(str.strip).tolist()
    else:
        tokens = [str(token).strip() for token in args.tokens if str(token).strip()]

    if len(tokens) == 0:
        utils.log("Annotate", LogType.ERROR, "No valid tokens found to annotate.")
        return

    def progress_callback(current, total, token, label, confidence):
        """Callback to log annotation progress at intervals (avoid spam for large datasets)"""
        # Log every 1000 tokens or every 5% progress, whichever is more frequent
        interval = max(1000, int(total * 0.05))

        if current % interval == 0 or current == total:
            percentage = (current / total) * 100
            utils.log(
                "Annotate",
                LogType.INFO,
                f"[{current}/{total} {percentage:5.1f}%] Processing...",
            )

    try:
        results, checkpoint_path = annotate_tokens(
            tokens=tokens,
            config_path=args.config,
            exp_id=args.exp_id,
            checkpoint_name=args.checkpoint_name,
            confidence_threshold=confidence_threshold,
            progress_callback=progress_callback if has_csv_arg else None,
            batch_size=64 if has_csv_arg else 1,
        )

        utils.log("Annotate", LogType.INFO, f"Using checkpoint: {checkpoint_path}")
        utils.log(
            "Annotate",
            LogType.INFO,
            f"Confidence threshold: {confidence_threshold:.4f}",
        )

        # Summary logging
        passed_count = sum(1 for row in results if row["passes_threshold"])
        unid_count = len(results) - passed_count
        utils.log(
            "Annotate",
            LogType.INFO,
            f"Annotation completed: {len(results)} total, {passed_count} passed threshold, {unid_count} UNID",
        )

        # If processing CSV without progress callback, show individual results
        if not has_csv_arg:
            for row in results:
                status = "✓" if row["passes_threshold"] else "✗"
                utils.log(
                    "Annotate",
                    LogType.INFO,
                    f"{status} {row['token']} -> {row['final_label']} (pred={row['predicted_label']}, conf={row['confidence']:.4f})",
                )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() == ".csv":
                if source_rows is not None:
                    annotated = source_rows.copy()
                    token_column = args.token_column

                    # Use pandas merge instead of lambda maps (much faster for large datasets)
                    results_df = pd.DataFrame(results)
                    results_df = results_df.rename(columns={"token": token_column})

                    # Merge on token column
                    annotated = annotated.merge(
                        results_df[
                            [
                                token_column,
                                "predicted_label",
                                "predicted_label_id",
                                "confidence",
                                "passes_threshold",
                                "final_label",
                            ]
                        ],
                        on=token_column,
                        how="left",
                    )

                    annotated.to_csv(output_path, index=False)
                else:
                    pd.DataFrame(results).to_csv(output_path, index=False)

            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

            utils.log("Annotate", LogType.INFO, f"Saved output: {output_path}")

    except Exception as error:
        utils.log("Annotate", LogType.ERROR, f"Annotation failed: {error}")

        raise


if __name__ == "__main__":
    main()
