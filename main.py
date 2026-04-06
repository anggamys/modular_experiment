from data import Data
from hugging_face import HuggingFace
from type import LogType
from utils import Utils


def main():
    data = Data()
    utils = Utils()
    hugging_face = HuggingFace()

    args = utils.argument_parser(
        description="A script to download models from Hugging Face.",
        arguments=[
            {
                "name": "--dataset_file",
                "help": "The path to the dataset file.",
                "required": True,
            },
            {
                "name": "--model_name",
                "help": "The name of the model to download from Hugging Face.",
                "required": True,
            },
        ],
    )

    try:
        df = data.load_data(args.dataset_file)

        utils.log(
            "Main",
            LogType.INFO,
            f"Loaded dataset from {args.dataset_file}. Shape: {df.shape}",
        )

        labels = df["final_pos_tag"].unique().tolist()
        label_id_map = data.label2id(labels)

        utils.log(
            "Main",
            LogType.INFO,
            f"Unique labels: {len(label_id_map)}. Label to ID mapping: {label_id_map}",
        )

        model_path = hugging_face.huggingface_download(args.model_name)

        utils.log(
            "Main",
            LogType.INFO,
            f"Model downloaded and available at: {model_path}",
        )

        tokenizer = hugging_face.tokenizer(model_path)
        model = hugging_face.model(model_path)

        hugging_face.model_info(model_path)

        sample_text = "Contoh kalimat untuk tokenisasi."
        token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        utils.log(
            "Main",
            LogType.INFO,
            f"Tokens: {tokens}. Token ID count: {len(token_ids)}",
        )

        if token_ids:
            embedding_vector = hugging_face.get_embedding_vector(model, token_ids[0])

            if embedding_vector is not None:
                utils.log(
                    "Main",
                    LogType.INFO,
                    f"Embedding vector for token '{tokens[0]}': {embedding_vector[:5]}... (truncated)",
                )

    except SystemExit:
        pass
    except Exception as e:
        utils.log("Main", LogType.ERROR, f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
