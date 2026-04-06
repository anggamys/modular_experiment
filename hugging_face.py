from typing import Optional

from utils import Utils, LogType
from huggingface_hub import snapshot_download


class HuggingFace:
    def __init__(self):
        self.utils = Utils()

    def huggingface_download(self, model_name: str, local_dir: Optional[str] = None):
        if local_dir is None:
            self.utils.log(
                "HuggingFace",
                LogType.INFO,
                f"No local directory provided. Using default: ./hugging_face/{model_name}",
            )
            local_dir = f"./hugging_face/{model_name}"

        self.utils.create_dir(local_dir)

        snapshot_download(repo_id=model_name, local_dir=local_dir)
        self.utils.log(
            "HuggingFace",
            LogType.INFO,
            f"Model '{model_name}' downloaded successfully to: {local_dir}",
        )

        return local_dir
