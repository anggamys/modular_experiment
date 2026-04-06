import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from type import LogType


class Utils:
    _log_file_path = None

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

            if Utils._log_file_path:
                with open(Utils._log_file_path, "a", encoding="utf-8") as f:
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

    def log2file(self, log_dir: str = "logs", filename: str | None = None):
        try:
            os.makedirs(log_dir, exist_ok=True)

            if filename is None:
                dt = self.dateTimeNow().replace(":", "-").replace(" ", "_")
                filename = f"run_{dt}.log"

            Utils._log_file_path = os.path.join(log_dir, filename)
            self.log("Utils", LogType.INFO, f"Log file enabled: {Utils._log_file_path}")

            return Utils._log_file_path
        except Exception as e:
            self.log("Utils", LogType.ERROR, f"Failed to enable log file: {e}")
            raise
