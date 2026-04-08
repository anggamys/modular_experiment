import argparse
import json
import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

from type import LogType


class Utils:
    _log_file_path = None
    _log_file_paths = []

    def __init__(self):
        pass

    def dateTimeNow(self) -> str:
        try:
            return datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

        except Exception:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_log(self, module: str, log_type: LogType, message: str) -> str:
        timestamp = self.dateTimeNow()
        return f"[{timestamp}] [{module}] [{log_type.value}]: {message}"

    def log(self, module: str, log_type: LogType, message: str) -> None:
        try:
            log_line = self._format_log(module, log_type, message)

            print(log_line)

            for log_file_path in Utils._log_file_paths:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")

        except Exception as e:
            print(f"[LOGGER ERROR] Failed to write log: {e}")

    def create_dir(self, path: str) -> str:
        try:
            if os.path.exists(path):
                self.log("Utils", LogType.INFO, f"Directory already exists: {path}")
                return path

            os.makedirs(path, exist_ok=True)
            self.log("Utils", LogType.INFO, f"Directory created: {path}")

            return path
        except Exception as e:
            self.log(
                "Utils", LogType.ERROR, f"Failed to create directory '{path}': {e}"
            )

            raise

    def argument_parser(
        self,
        description: str,
        arguments: list,
    ):
        try:
            parser = argparse.ArgumentParser(description=description)

            for arg in arguments:
                arg_copy = dict(arg)
                name = arg_copy.pop("name")
                help_text = arg_copy.pop("help", None)
                if help_text is not None:
                    arg_copy["help"] = help_text
                parser.add_argument(name, **arg_copy)

            return parser.parse_args()

        except SystemExit:
            raise

        except Exception as e:
            self.log("Utils", LogType.ERROR, f"Argument parsing failed: {e}")
            raise

    def has_file_logging(self) -> bool:
        return len(Utils._log_file_paths) > 0

    def log2file(self, log_dir: str = "logs", filename: str | None = None):
        try:
            os.makedirs(log_dir, exist_ok=True)

            if filename is None:
                dt = self.dateTimeNow().replace(":", "-").replace(" ", "_")
                filename = f"run_{dt}.log"

            log_file_path = os.path.join(log_dir, filename)
            Utils._log_file_path = log_file_path

            if log_file_path not in Utils._log_file_paths:
                Utils._log_file_paths.append(log_file_path)

            self.log("Utils", LogType.INFO, f"Log file enabled: {log_file_path}")

            return log_file_path
        except Exception as e:
            self.log("Utils", LogType.ERROR, f"Failed to enable log file: {e}")
            raise

    def write_json(self, path, payload, ensure_ascii: bool = False):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=ensure_ascii)
        except Exception as e:
            self.log("Utils", LogType.ERROR, f"Failed to write JSON '{path}': {e}")
            raise

    def setup_runtime(self):
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
        try:
            from huggingface_hub.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:
            # Runtime setup should be best effort.
            pass
