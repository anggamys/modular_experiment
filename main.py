import argparse

from data import Data
from hugging_face import HuggingFace
from type import LogType
from utils import Utils


def test_embedding(utils, data, hugging_face, model_name):
    """Test embedding extraction from model."""
    utils.log("Main", LogType.INFO, "Running embedding test...")

    try:
        model_path = hugging_face.huggingface_download(model_name)

        utils.log(
            "Main",
            LogType.INFO,
            f"Model downloaded and available at: {model_path}",
        )

        tokenizer = hugging_face.tokenizer(model_path)
        model = hugging_face.model(model_path)

        hugging_face.model_info(model_path)

        sample_text = "Contoh kalimat untuk tokenisasi."
        token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        utils.log(
            "Main",
            LogType.INFO,
            f"Tokens: {tokens}. Token ID count: {len(token_ids)}",
        )

        if token_ids:
            embedding_vector = hugging_face.get_embedding_vector(model, token_ids[0])

            if embedding_vector is not None:
                utils.log(
                    "Main",
                    LogType.INFO,
                    f"Embedding vector for token '{tokens[0]}': {embedding_vector[:5]}... (truncated)",
                )

        utils.log("Main", LogType.INFO, "Embedding test completed successfully!")

    except SystemExit:
        pass
    except Exception as e:
        utils.log("Main", LogType.ERROR, f"Embedding test failed: {e}")
        raise


def explore_data(utils, data, dataset_file):
    """Explore dataset structure."""
    utils.log("Main", LogType.INFO, f"Loading and exploring dataset: {dataset_file}")

    try:
        df = data.load_data(dataset_file)

        utils.log(
            "Main",
            LogType.INFO,
            f"Loaded dataset from {dataset_file}. Shape: {df.shape}",
        )

        labels = df["final_pos_tag"].unique().tolist()
        label_id_map = data.label2id(labels)

        utils.log(
            "Main",
            LogType.INFO,
            f"Unique labels: {len(label_id_map)}",
        )

        utils.log(
            "Main",
            LogType.INFO,
            f"Label to ID mapping: {label_id_map}",
        )

        # Show sample data
        utils.log(
            "Main",
            LogType.INFO,
            f"Sample data (first 5 rows):\n{df.head()}",
        )

        utils.log("Main", LogType.INFO, "Data exploration completed!")

    except SystemExit:
        pass
    except Exception as e:
        utils.log("Main", LogType.ERROR, f"Data exploration failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="IndoBERT POS Tagging Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore dataset
  python main.py --mode explore --dataset_file data.csv

  # Test embedding extraction
  python main.py --mode embed --model_name indolem/indobert-base-p1

  # Start training
  python train.py --dataset data.csv --config config.yml
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["explore", "embed"],
        default="explore",
        help="Mode to run: explore (data exploration) or embed (test embeddings)",
    )

    parser.add_argument(
        "--dataset_file",
        type=str,
        help="Path to dataset CSV file (required for explore mode)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="indolem/indobert-base-p1",
        help="Model name from HuggingFace (default: indolem/indobert-base-p1)",
    )

    parser.add_argument(
        "--log_file",
        action="store_true",
        help="Enable logging to file",
    )

    args = parser.parse_args()

    utils = Utils()
    data = Data()
    hugging_face = HuggingFace()

    if args.log_file:
        utils.log2file()

    utils.log("Main", LogType.INFO, f"Starting in {args.mode} mode")

    try:
        if args.mode == "explore":
            if not args.dataset_file:
                parser.error("--dataset_file is required for explore mode")
            explore_data(utils, data, args.dataset_file)

        elif args.mode == "embed":
            test_embedding(utils, data, hugging_face, args.model_name)

    except SystemExit:
        pass
    except Exception as e:
        utils.log("Main", LogType.ERROR, f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
