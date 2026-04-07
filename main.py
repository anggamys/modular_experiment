from utils import Utils
from type import LogType
from train import Trainer


def main():
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

        (
            train_dataset,
            val_dataset,
            test_dataset,
            label2id,
            id2label,
            model_path,
        ) = trainer.prepare_data(args.dataset)

        model, label2id, id2label = trainer.train(
            train_dataset, val_dataset, label2id, id2label, model_path
        )

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
