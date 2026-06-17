#!/usr/bin/env python3
"""Check whether a local model cache directory is ready for inference.

This command inspects files only. It does not import ML libraries, load model
weights, initialize devices, or touch the network.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SHARD_PATTERN = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors$")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
)


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def detect_shards(model_dir: Path) -> tuple[int | None, list[Path], list[Path]]:
    shard_files: list[Path] = []
    malformed: list[Path] = []
    expected_total: int | None = None

    for path in sorted(model_dir.glob("model-*-of-*.safetensors")):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None:
            malformed.append(path)
            continue
        shard_files.append(path)
        total = int(match.group(2))
        if expected_total is None:
            expected_total = total
        elif expected_total != total:
            malformed.append(path)

    return expected_total, shard_files, malformed


def missing_shard_names(expected_total: int, shards: list[Path]) -> list[str]:
    present = set()
    for path in shards:
        match = SHARD_PATTERN.fullmatch(path.name)
        if match:
            present.add(int(match.group(1)))
    return [f"model-{index:05d}-of-{expected_total:05d}.safetensors" for index in range(1, expected_total + 1) if index not in present]


def has_tokenizer_file(model_dir: Path) -> bool:
    if any((model_dir / filename).is_file() for filename in TOKENIZER_FILES):
        return True
    return (model_dir / "merges.txt").is_file() and (model_dir / "vocab.json").is_file()


def check_model_cache(model_dir: Path, expected_shards: int | None, allow_missing_tokenizer: bool) -> int:
    resolved = resolve_path(model_dir)
    problems: list[str] = []

    print("Model cache preflight")
    print(f"Model directory: {resolved}")

    if not resolved.is_dir():
        print("Status: not ready")
        print("Problem: model directory does not exist")
        return 1

    config_path = resolved / "config.json"
    if not config_path.is_file():
        problems.append("missing config.json")

    if not allow_missing_tokenizer and not has_tokenizer_file(resolved):
        tokenizer_options = ", ".join(TOKENIZER_FILES)
        problems.append(f"missing tokenizer file ({tokenizer_options}, or vocab.json plus merges.txt)")

    detected_total, shards, malformed = detect_shards(resolved)
    total = expected_shards or detected_total
    zero_byte_shards = [path.name for path in shards if path.stat().st_size == 0]

    if malformed:
        problems.append("malformed or inconsistent shard filenames: " + ", ".join(path.name for path in malformed))
    if total is None:
        problems.append("no model shard files matching model-00001-of-000NN.safetensors")
    else:
        missing = missing_shard_names(total, shards)
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... ({len(missing)} total)"
            problems.append(f"missing shard files: {preview}")
        if zero_byte_shards:
            problems.append("zero-byte shard files: " + ", ".join(zero_byte_shards))

    print(f"Config present: {config_path.is_file()}")
    print(f"Tokenizer present: {has_tokenizer_file(resolved)}")
    print(f"Detected shards: {len(shards)}")
    print(f"Expected shards: {total if total is not None else 'unknown'}")
    print(f"Zero-byte shards: {len(zero_byte_shards)}")
    print("Model loading: not attempted")
    print("Network access: not attempted")

    if problems:
        print("Status: not ready")
        for problem in problems:
            print(f"Problem: {problem}")
        return 1

    print("Status: ready for an inference dry run")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Local model cache directory to inspect.")
    parser.add_argument("--expected-shards", type=int, help="Expected number of safetensors shard files.")
    parser.add_argument(
        "--allow-missing-tokenizer",
        action="store_true",
        help="Do not fail when tokenizer files are absent.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return check_model_cache(
        model_dir=args.model_dir,
        expected_shards=args.expected_shards,
        allow_missing_tokenizer=args.allow_missing_tokenizer,
    )


if __name__ == "__main__":
    raise SystemExit(main())
