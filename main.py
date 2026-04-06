import argparse
from utils import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Name of the model to download")
    args = parser.parse_args()

    if args.model_name:
        snapshot_download(args.model_name)
    else:
        print("Please provide a model name to download.")

if __name__ == "__main__":
    main()