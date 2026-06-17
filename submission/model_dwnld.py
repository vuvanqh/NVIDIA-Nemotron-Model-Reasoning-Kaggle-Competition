import argparse
import os
from pathlib import Path


MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
DEFAULT_CACHE_DIR = Path(r"E:\kagglehub")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Nemotron base model via KaggleHub.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"KaggleHub cache directory. Defaults to {DEFAULT_CACHE_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve(strict=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    import kagglehub

    path = kagglehub.model_download(MODEL_HANDLE)
    print(f"KaggleHub cache: {cache_dir}")
    print(path)


if __name__ == "__main__":
    main()
