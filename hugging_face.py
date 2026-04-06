from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from utils import LogType, Utils


class HuggingFace:
    def __init__(self):
        self.utils = Utils()

    def _resolve_local_dir(self, model_name: str, local_dir: Optional[str]) -> str:
        if local_dir is None:
            local_dir = str(Path("./hugging_face") / model_name)
            self.utils.log(
                "HuggingFace",
                LogType.INFO,
                f"No local directory provided. Using default: {local_dir}",
            )

        return local_dir

    def huggingface_download(
        self, model_name: str, local_dir: Optional[str] = None
    ) -> str:
        local_dir = self._resolve_local_dir(model_name, local_dir)
        local_path = Path(local_dir)

        if local_path.exists() and any(local_path.iterdir()):
            self.utils.log(
                "HuggingFace",
                LogType.INFO,
                f"Model '{model_name}' already exists. Using local folder: {local_dir}",
            )
            return local_dir

        self.utils.create_dir(local_dir)

        try:
            snapshot_download(repo_id=model_name, local_dir=local_dir)
        except Exception as error:
            self.utils.log(
                "HuggingFace",
                LogType.ERROR,
                f"Failed to download model '{model_name}': {error}",
            )

            exit(1)

        self.utils.log(
            "HuggingFace",
            LogType.INFO,
            f"Model '{model_name}' downloaded successfully to: {local_dir}",
        )

        return local_dir

    def tokenizer(self, model_path: str) -> PreTrainedTokenizerBase:
        if model_path is None:
            self.utils.log(
                "HuggingFace",
                LogType.ERROR,
                "No model path provided for tokenizer.",
            )

            exit(1)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as error:
            self.utils.log(
                "HuggingFace",
                LogType.ERROR,
                f"Failed to load tokenizer from '{model_path}': {error}",
            )

            exit(1)

        return tokenizer

    def model(self, model_path: str) -> PreTrainedModel:
        if model_path is None:
            self.utils.log(
                "HuggingFace",
                LogType.ERROR,
                "No model path provided for model loading.",
            )

            exit(1)

        try:
            model = AutoModel.from_pretrained(model_path)
        except Exception as error:
            self.utils.log(
                "HuggingFace",
                LogType.ERROR,
                f"Failed to load model from '{model_path}': {error}",
            )

            exit(1)

        return model

    def get_embedding_vector(self, model: PreTrainedModel, token_id: int):
        try:
            embeddings = model.get_input_embeddings()
            if embeddings is None:
                self.utils.log(
                    "HuggingFace",
                    LogType.ERROR,
                    "Model does not provide input embeddings.",
                )

                return None

            return embeddings.weight[token_id].tolist()
        except Exception as error:
            self.utils.log(
                "HuggingFace",
                LogType.ERROR,
                f"Failed to get embedding vector for token ID {token_id}: {error}",
            )

            return None
