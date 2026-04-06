import os
from typing import Optional

from huggingface_hub import snapshot_download


class Utils:
    def __init__(self):
        pass

    def create_dir(self, path: str):
        if os.path.exists(path):
            return

        if not os.path.exists(path):
            os.makedirs(path)
            
        return path

    def huggingface_download(self, model_name: str, local_dir: Optional[str] = None):
        if local_dir is None:
            local_dir = f"./hugging_face/{model_name}"

        self.create_dir(local_dir)

        snapshot_download(repo_id=model_name, local_dir=local_dir)
        
        return local_dir