import enum
import os


class LogType(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Utils:
    def __init__(self):
        pass

    def log(self, module: str, log_type: LogType, message: str):
        print(f"[{module}] {log_type.value}: {message}")

    def create_dir(self, path: str):
        if os.path.exists(path):
            self.log("Utils", LogType.INFO, f"Directory already exists: {path}")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        self.log("Utils", LogType.INFO, f"Directory created: {path}")
        return path
