from utils import Utils
from type import LogType
from main import run_experiment


VALID_EXPERIMENTS = {"EXP-07", "EXP-08", "EXP-09"}


def main():
    utils = Utils()
    args = utils.argument_parser(
        description="Char-level experiments",
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
                "help": "EXP-07 to EXP-09",
                "type": str,
                "default": "EXP-07",
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

    if args.exp_id not in VALID_EXPERIMENTS:
        utils.log(
            "MainChar",
            LogType.ERROR,
            f"Invalid exp_id={args.exp_id}. Allowed: {sorted(VALID_EXPERIMENTS)}",
        )

        raise SystemExit(1)

    run_experiment(args.dataset, args.config, args.exp_id, args.log_file)


if __name__ == "__main__":
    main()
