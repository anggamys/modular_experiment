import argparse

from utils import Utils, LogType
from hugging_face import HuggingFace


def main():
    utils = Utils()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Name of the model to download")
    args = parser.parse_args()

    hugging_face = HuggingFace()

    if args.model_name:
        hugging_face.huggingface_download(args.model_name)
    else:
        utils.log(
            "Main",
            LogType.WARNING,
            "No model name provided. Please use --model_name to specify the model to download.",
        )


if __name__ == "__main__":
    main()
