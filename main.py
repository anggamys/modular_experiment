from utils import Utils
from type import LogType
from data import DataPipeline
from model import ModelBuilder
from hugging_face import HuggingFace
from train import Trainer
import os
import warnings
from huggingface_hub.utils import logging as hf_logging


def run_experiment(dataset: str, config_path: str, exp_id: str, log_file: bool):
    utils = Utils()

    if log_file:
        utils.log2file()

    utils.log(
        "Main",
        LogType.INFO,
        f"Config: {config_path} | Dataset: {dataset} | Exp: {exp_id}",
    )

    trainer = Trainer(config_path, exp_id=exp_id)
    data_pipeline = DataPipeline()
    model_builder = ModelBuilder()
    hugging_face = HuggingFace()

    architecture = trainer.config["model"]["architecture"]
    use_transformer = architecture in {
        "bert_linear",
        "bert_mlp",
        "bert_gru",
        "bert_cnn",
        "hybrid_bert_charcnn",
    }

    tokenizer = None
    bert_model = None

    if use_transformer:
        model_path = hugging_face.huggingface_download(
            trainer.config["model"]["model_name"]
        )
        tokenizer = hugging_face.tokenizer(model_path)
        bert_model = hugging_face.model(model_path)

    train_dataset, val_dataset, test_dataset, label2id, id2label, metadata = (
        data_pipeline.prepare_datasets(
            csv_path=dataset,
            tokenizer=tokenizer,
            test_size=trainer.config["data"]["test_size"],
            validation_size=trainer.config["data"]["validation_size"],
            random_state=trainer.config["data"]["random_state"],
            max_length=trainer.config["data"]["max_length"],
            architecture=architecture,
            label_column=trainer.config["data"].get("label_column", "label"),
            char_max_length=trainer.config["data"].get("char_max_length", 32),
        )
    )

    model = model_builder.build_model(
        config_model=trainer.config["model"],
        num_labels=len(label2id),
        bert_model=bert_model,
        char_vocab_size=metadata.get("char_vocab_size"),
    )

    model = trainer.train(model, train_dataset, val_dataset, label2id, id2label)
    trainer.evaluate(model, test_dataset, id2label)

    utils.log(
        "Main",
        LogType.INFO,
        f"Done! Results: {trainer.checkpoint_dir}",
    )


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
        description="Single-token POS research runner",
        arguments=[
            {
                "name": "--dataset",
                "help": "Path to dataset CSV",
                "type": str,
                "required": True,
            },
            {
                "name": "--config",
                "help": "Path to config file",
                "type": str,
                "default": "config.yml",
                "required": False,
            },
            {
                "name": "--exp_id",
                "help": "Experiment ID (EXP-01 ... EXP-11)",
                "type": str,
                "default": "EXP-01",
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

    try:
        run_experiment(args.dataset, args.config, args.exp_id, args.log_file)

    except SystemExit:
        pass

    except Exception as e:
        utils.log("Main", LogType.ERROR, f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
