import os
import argparse

from type import LogType
from datetime import datetime
from zoneinfo import ZoneInfo


class Utils:
    def __init__(self):
        pass

    def log(self, module: str, log_type: LogType, message: str):
        timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{timestamp}] [{module}] {log_type.value}: {message}")

    def create_dir(self, path: str):
        if os.path.exists(path):
            self.log("Utils", LogType.INFO, f"Directory already exists: {path}")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        self.log("Utils", LogType.INFO, f"Directory created: {path}")
        return path

    def argument_parser(
        self,
        description: str,
        arguments: list,
    ):
        parser = argparse.ArgumentParser(description=description)

        for arg in arguments:
            parser.add_argument(
                arg["name"], help=arg["help"], required=arg.get("required", False)
            )

        return parser.parse_args()
