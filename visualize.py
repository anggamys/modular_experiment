"""
Script untuk visualisasi training results dan evaluation metrics.
"""

import argparse
import json
from pathlib import Path
import yaml


def plot_training_curves(results_path: str):
    """
    Plot training curves dari training results.

    Args:
        results_path: Path ke training_results.json
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    print("Training Results Summary:")

    epochs = results.get("epochs", [])
    train_losses = results.get("train_loss", [])
    val_losses = results.get("val_loss", [])
    val_accs = results.get("val_accuracy", [])
    val_f1s = results.get("val_f1", [])

    if not epochs:
        print("No training results found!")
        return

    print(f"Total Epochs: {len(epochs)}\n")

    print("Epoch | Train Loss | Val Loss | Val Accuracy | Val F1-Score")

    for i, epoch in enumerate(epochs):
        train_loss = train_losses[i] if i < len(train_losses) else "-"
        val_loss = val_losses[i] if i < len(val_losses) else "-"
        val_acc = val_accs[i] if i < len(val_accs) else "-"
        val_f1 = val_f1s[i] if i < len(val_f1s) else "-"

        if isinstance(train_loss, (int, float)):
            train_loss = f"{train_loss:.4f}"
        if isinstance(val_loss, (int, float)):
            val_loss = f"{val_loss:.4f}"
        if isinstance(val_acc, (int, float)):
            val_acc = f"{val_acc:.4f}"
        if isinstance(val_f1, (int, float)):
            val_f1 = f"{val_f1:.4f}"

        print(
            f"{epoch:5} | {train_loss:10} | {val_loss:8} | {val_acc:12} | {val_f1:12}"
        )

    # Summary
    print("\n")
    if train_losses:
        min_train_loss = min([x for x in train_losses if isinstance(x, (int, float))])
        print(f"Best Train Loss: {min_train_loss:.4f}")

    if val_losses:
        min_val_loss = min([x for x in val_losses if isinstance(x, (int, float))])
        best_epoch_loss = val_losses.index(min_val_loss) + 1
        print(f"Best Val Loss: {min_val_loss:.4f} (Epoch {best_epoch_loss})")

    if val_accs:
        max_val_acc = max([x for x in val_accs if isinstance(x, (int, float))])
        best_epoch_acc = val_accs.index(max_val_acc) + 1
        print(f"Best Val Accuracy: {max_val_acc:.4f} (Epoch {best_epoch_acc})")

    if val_f1s:
        max_val_f1 = max([x for x in val_f1s if isinstance(x, (int, float))])
        best_epoch_f1 = val_f1s.index(max_val_f1) + 1
        print(f"Best Val F1-Score: {max_val_f1:.4f} (Epoch {best_epoch_f1})")

    print("\n")


def print_evaluation_report(eval_path: str):
    """
    Print evaluation report dari evaluation results.

    Args:
        eval_path: Path ke evaluation_results.json
    """
    with open(eval_path, "r") as f:
        results = json.load(f)

    print("\n")
    print("Evaluation Results (Test Set)")
    print("\n")

    print("Metrics Summary:")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("\n")


def print_config(config_path: str):
    """
    Print training configuration.

    Args:
        config_path: Path ke config.yml
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("\n")
    print("Training Configuration")
    print("\n")

    print("Model Configuration:")
    model = config.get("model", {})
    print(f"  Model Name: {model.get('model_name')}")
    print(f"  Hidden Size: {model.get('hidden_size')}")
    print(f"  Freeze BERT: {model.get('freeze_bert')}")

    print("\nTraining Configuration:")
    training = config.get("training", {})
    print(f"  Batch Size: {training.get('batch_size')}")
    print(f"  Learning Rate: {training.get('learning_rate')}")
    print(f"  Num Epochs: {training.get('num_epochs')}")
    print(f"  Weight Decay: {training.get('weight_decay')}")
    print(f"  Warmup Steps: {training.get('warmup_steps')}")

    print("\nData Configuration:")
    data = config.get("data", {})
    print(f"  Test Size: {data.get('test_size')}")
    print(f"  Validation Size: {data.get('validation_size')}")
    print(f"  Max Length: {data.get('max_length')}")

    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualisasi training dan evaluation results"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Path to checkpoint directory",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./config.yml",
        help="Path to config file",
    )

    parser.add_argument(
        "--training",
        action="store_true",
        help="Show training results",
    )

    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Show evaluation results",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all results",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    # Show config if specified
    if Path(args.config).exists():
        print_config(args.config)

    if args.all or args.training:
        training_results = checkpoint_dir / "training_results.json"
        if training_results.exists():
            plot_training_curves(str(training_results))
        else:
            print("training_results.json not found!")

    if args.all or args.evaluation:
        eval_results = checkpoint_dir / "evaluation_results.json"
        if eval_results.exists():
            print_evaluation_report(str(eval_results))
        else:
            print("evaluation_results.json not found!")

    if not (args.all or args.training or args.evaluation):
        # Default: show all
        training_results = checkpoint_dir / "training_results.json"
        eval_results = checkpoint_dir / "evaluation_results.json"

        if training_results.exists():
            plot_training_curves(str(training_results))

        if eval_results.exists():
            print_evaluation_report(str(eval_results))

        if not training_results.exists() and not eval_results.exists():
            print("No results found in checkpoint directory!")


if __name__ == "__main__":
    main()
