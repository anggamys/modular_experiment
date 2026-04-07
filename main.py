from utils import Utils
from type import LogType
from data import DataPipeline
from model import ModelBuilder
from hugging_face import HuggingFace
from train import Trainer
import os
import warnings
from huggingface_hub.utils import logging as hf_logging


def main():
    # Keep logs focused on app logs by muting noisy third-party warnings.
    warnings.filterwarnings(
        "ignore",
        message="This DataLoader will create .* worker processes.*",
        category=UserWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message=r".*GradScaler.*deprecated.*",
        category=FutureWarning,
    )

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()

    utils = Utils()
    args = utils.argument_parser(
        description="IndoBERT POS Tagging Training",
        arguments=[
            {
                "name": "--dataset",
                "help": "Path to dataset CSV",
                "type": str,
                "required": True,
            },
            {
                "name": "--config",
                "help": "Path to config file (default: config.yml)",
                "type": str,
                "default": "config.yml",
                "required": False,
            },
            {
                "name": "--log_file",
                "help": "Enable file logging",
                "action": "store_true",
                "required": False,
            },
        ],
    )

    if args.log_file:
        utils.log2file()

    utils.log("Main", LogType.INFO, f"Config: {args.config} | Dataset: {args.dataset}")

    try:
        trainer = Trainer(args.config)
        data_pipeline = DataPipeline()
        model_builder = ModelBuilder()
        hugging_face = HuggingFace()

        model_path = hugging_face.huggingface_download(
            trainer.config["model"]["model_name"]
        )
        tokenizer = hugging_face.tokenizer(model_path)

        train_dataset, val_dataset, test_dataset, label2id, id2label = (
            data_pipeline.prepare_datasets(
                csv_path=args.dataset,
                tokenizer=tokenizer,
                test_size=trainer.config["data"]["test_size"],
                validation_size=trainer.config["data"]["validation_size"],
                random_state=trainer.config["data"]["random_state"],
                max_length=trainer.config["data"]["max_length"],
            )
        )

        bert_model = hugging_face.model(model_path)
        model = model_builder.build_token_classifier(
            bert_model=bert_model,
            num_labels=len(label2id),
            hidden_size=trainer.config["model"]["hidden_size"],
            freeze_bert=trainer.config["model"]["freeze_bert"],
        )

        model = trainer.train(model, train_dataset, val_dataset, label2id, id2label)

        trainer.evaluate(model, test_dataset, id2label)

        utils.log(
            "Main",
            LogType.INFO,
            f"Done! Results: {trainer.checkpoint_dir}",
        )

    except SystemExit:
        pass

    except Exception as e:
        utils.log("Main", LogType.ERROR, f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
