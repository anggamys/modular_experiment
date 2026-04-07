from utils import Utils
from type import LogType
from main import run_experiment


VALID_EXPERIMENTS = {"EXP-10", "EXP-11"}


def main():
    utils = Utils()
    args = utils.argument_parser(
        description="Hybrid transformer + char-level experiments",
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
                "help": "EXP-10 to EXP-11",
                "type": str,
                "default": "EXP-10",
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
            "MainHybrid",
            LogType.ERROR,
            f"Invalid exp_id={args.exp_id}. Allowed: {sorted(VALID_EXPERIMENTS)}",
        )
        raise SystemExit(1)

    run_experiment(args.dataset, args.config, args.exp_id, args.log_file)


if __name__ == "__main__":
    main()
