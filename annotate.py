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

    model.load_state_dict(payload["model_state_dict"])
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

    with torch.no_grad():
        for token in tokens:
            token = str(token).strip()
            if token == "":
                continue

            batch = {}
            if architecture in {
                "bert_linear",
                "bert_mlp",
                "bert_gru",
                "bert_cnn",
            }:
                batch.update(_build_token_batch(token, tokenizer, max_length))
            elif architecture in {"char_cnn", "char_bilstm", "char_cnn_bilstm"}:
                batch.update(_build_char_batch(token, char_vocab, char_max_length))
            elif architecture == "hybrid_bert_charcnn":
                batch.update(_build_token_batch(token, tokenizer, max_length))
                batch.update(_build_char_batch(token, char_vocab, char_max_length))
            else:
                raise RuntimeError(
                    f"Unsupported architecture for annotation: {architecture}"
                )

            batch = {
                key: value.to(trainer.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type=trainer.device.type,
                enabled=trainer.device.type == "cuda",
            ):
                model_out = model(**batch)

            logits = model_out["logits"]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, pred_idx].item())
            label = id2label.get(pred_idx, str(pred_idx))

            outputs.append(
                {
                    "token": token,
                    "predicted_label": label,
                    "predicted_label_id": pred_idx,
                    "confidence": confidence,
                }
            )

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
                "type": str,
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
                "name": "--output",
                "help": "Optional JSON output file path",
                "type": str,
                "required": False,
            },
        ],
    )

    has_tokens_arg = bool(args.tokens and str(args.tokens).strip())
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
        tokens = [token for token in args.tokens.split() if token.strip()]

    if len(tokens) == 0:
        utils.log("Annotate", LogType.ERROR, "No valid tokens found to annotate.")
        return

    try:
        results, checkpoint_path = annotate_tokens(
            tokens=tokens,
            config_path=args.config,
            exp_id=args.exp_id,
            checkpoint_name=args.checkpoint_name,
        )

        utils.log("Annotate", LogType.INFO, f"Using checkpoint: {checkpoint_path}")

        for row in results:
            utils.log(
                "Annotate",
                LogType.INFO,
                f"{row['token']} -> {row['predicted_label']} (conf={row['confidence']:.4f})",
            )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() == ".csv":
                if source_rows is not None:
                    annotated = source_rows.copy()
                    token_column = args.token_column
                    pred_map = {row["token"]: row for row in results}
                    annotated["predicted_label"] = (
                        annotated[token_column]
                        .astype(str)
                        .map(
                            lambda x: pred_map.get(x.strip(), {}).get("predicted_label")
                        )
                    )
                    annotated["predicted_label_id"] = (
                        annotated[token_column]
                        .astype(str)
                        .map(
                            lambda x: pred_map.get(x.strip(), {}).get(
                                "predicted_label_id"
                            )
                        )
                    )
                    annotated["confidence"] = (
                        annotated[token_column]
                        .astype(str)
                        .map(lambda x: pred_map.get(x.strip(), {}).get("confidence"))
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
