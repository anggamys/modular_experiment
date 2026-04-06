"""
Example script untuk menggunakan pipeline secara programmatik.
Ini berguna untuk integration dengan workflow lain atau automation.
"""

import json
from pathlib import Path

from inference import Inference
from train import Trainer


def example_1_simple_training():
    """
    Example 1: Training sederhana dengan default config.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Training")
    print("=" * 80)

    # Initialize trainer
    trainer = Trainer("config.yml")

    # Prepare data
    dataset_path = "../pos-tagging/token/result/tabular/final_validation_data.csv"
    train_dataset, val_dataset, test_dataset, label2id, id2label = trainer.prepare_data(
        dataset_path
    )

    # Get model path
    model_path = trainer.hugging_face.huggingface_download(
        trainer.config["model"]["model_name"]
    )

    # Train
    model, label2id, id2label, training_results = trainer.train(
        train_dataset, val_dataset, label2id, id2label, model_path
    )

    # Evaluate
    eval_results = trainer.evaluate(model, test_dataset, id2label)

    print("\nTraining completed!")
    print(f"Best model saved to: {trainer.checkpoint_dir}")
    print(f"Training results: {json.dumps(training_results, indent=2)}")
    print(f"Evaluation results: {json.dumps(eval_results, indent=2)}")


def example_2_custom_config_training():
    """
    Example 2: Training dengan custom configuration.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Config Training")
    print("=" * 80)

    # Load default config
    import yaml

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Modify config for this experiment
    config["training"]["batch_size"] = 16
    config["training"]["num_epochs"] = 3
    config["training"]["learning_rate"] = 1e-5

    # Save modified config
    custom_config_path = "config_custom.yml"
    with open(custom_config_path, "w") as f:
        yaml.dump(config, f)

    # Use custom config
    trainer = Trainer(custom_config_path)

    # ... rest of training code similar to example 1
    print(f"Using custom config: {custom_config_path}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")


def example_3_transfer_learning():
    """
    Example 3: Transfer learning dengan frozen BERT encoder.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Transfer Learning (Frozen BERT)")
    print("=" * 80)

    import yaml

    # Load and modify config to freeze BERT
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    config["model"]["freeze_bert"] = True
    config["training"]["learning_rate"] = 1e-3  # Higher LR for linear probe

    # Save config
    transfer_config_path = "config_transfer.yml"
    with open(transfer_config_path, "w") as f:
        yaml.dump(config, f)

    trainer = Trainer(transfer_config_path)
    print("Transfer learning mode: BERT encoder frozen")
    print(f"Linear probe learning rate: {config['training']['learning_rate']}")


def example_4_batch_inference():
    """
    Example 4: Batch inference dengan trained model.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Inference")
    print("=" * 80)

    # Path to trained model
    checkpoint_path = "checkpoints/model_epoch_1.pt"

    # Initialize inference
    inference = Inference(checkpoint_path)

    # List of tokens to predict
    tokens = ["0", "1", "3", "97536", "12"]

    # Batch prediction
    results = inference.predict_batch(tokens)

    # Print results
    print(f"\nPredictions for {len(tokens)} tokens:")
    print("-" * 70)

    for result in results:
        token = result["token"]
        label = result["predicted_label"]
        confidence = result["confidence"]
        print(
            f"Token: {token:10} → Predicted: {label:10} (confidence: {confidence:.4f})"
        )

    print("-" * 70)


def example_5_load_and_analyze_results():
    """
    Example 5: Load dan analyze training results.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Loading & Analyzing Results")
    print("=" * 80)

    checkpoint_dir = Path("checkpoints")

    # Load training results
    training_results_path = checkpoint_dir / "training_results.json"
    if training_results_path.exists():
        with open(training_results_path, "r") as f:
            training_results = json.load(f)

        print("\nTraining Metrics:")
        print(f"  Epochs: {training_results['epochs']}")
        print(f"  Final Train Loss: {training_results['train_loss'][-1]:.4f}")
        print(f"  Final Val Loss: {training_results['val_loss'][-1]:.4f}")
        print(f"  Best Val Accuracy: {max(training_results['val_accuracy']):.4f}")
        print(f"  Best Val F1: {max(training_results['val_f1']):.4f}")

    # Load evaluation results
    eval_results_path = checkpoint_dir / "evaluation_results.json"
    if eval_results_path.exists():
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)

        print("\nEvaluation Metrics (Test Set):")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall: {eval_results['recall']:.4f}")
        print(f"  F1-Score: {eval_results['f1']:.4f}")


def example_6_compare_models():
    """
    Example 6: Compare multiple model checkpoints.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Compare Model Checkpoints")
    print("=" * 80)

    checkpoint_dir = Path("checkpoints")

    # Find all model checkpoints
    model_files = sorted(checkpoint_dir.glob("model_epoch_*.pt"))

    print(f"\nFound {len(model_files)} model checkpoints:")

    for model_file in model_files:
        # Load checkpoint
        import torch

        checkpoint = torch.load(model_file, map_location="cpu")

        epoch = checkpoint.get("epoch", "?")
        loss = checkpoint.get("loss", "?")

        print(f"  - {model_file.name}: Epoch {epoch}, Val Loss: {loss:.4f}")

    # Find best model
    if model_files:
        print(f"\nBest model: {model_files[0].name}")
        print(f"Use for inference: python inference.py --checkpoint {model_files[0]}")


def main():
    """Run examples."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python examples.py <example_number>")
        print("  1: Simple Training")
        print("  2: Custom Config Training")
        print("  3: Transfer Learning (Frozen BERT)")
        print("  4: Batch Inference")
        print("  5: Load & Analyze Results")
        print("  6: Compare Model Checkpoints")
        print("\nExample: python examples.py 1")
        return

    example_num = sys.argv[1]

    examples = {
        "1": example_1_simple_training,
        "2": example_2_custom_config_training,
        "3": example_3_transfer_learning,
        "4": example_4_batch_inference,
        "5": example_5_load_and_analyze_results,
        "6": example_6_compare_models,
    }

    if example_num in examples:
        examples[example_num]()
    else:
        print(f"Example {example_num} not found!")
        print(f"Available: {', '.join(examples.keys())}")


if __name__ == "__main__":
    main()
