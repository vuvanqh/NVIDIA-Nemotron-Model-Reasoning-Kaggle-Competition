#!/usr/bin/env python3
"""Package a real LoRA adapter directory into a submission zip."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


REQUIRED_CONFIG = "adapter_config.json"
MAX_LORA_RANK = 32
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
ALLOWED_SUFFIXES = {
    ".json",
    ".safetensors",
    ".bin",
    ".md",
    ".txt",
    ".model",
    ".jsonl",
}
EXCLUDED_PARTS = {
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "checkpoint",
    "checkpoints",
    "logs",
    "runs",
    "wandb",
    "data",
    "datasets",
    "cache",
}


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def ensure_output_under_submission_zip(path: Path) -> Path:
    output_path = resolve_path(path)
    package_root = (submission_root() / "submission_zip").resolve()
    try:
        output_path.relative_to(package_root)
    except ValueError:
        raise ValueError(f"--output-zip must be under {package_root}") from None
    if output_path.suffix != ".zip":
        raise ValueError("--output-zip must end with .zip")
    return output_path


def validate_packaging_inputs(adapter_dir: Path) -> tuple[list[Path], list[str]]:
    problems: list[str] = []
    if not adapter_dir.is_dir():
        return [], [f"adapter directory not found: {adapter_dir}"]

    config_path = adapter_dir / REQUIRED_CONFIG
    if not config_path.is_file():
        problems.append(f"missing required file: {REQUIRED_CONFIG}")
    else:
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except json.JSONDecodeError as exc:
            problems.append(f"invalid JSON in {REQUIRED_CONFIG}: {exc}")
        else:
            if not isinstance(config, dict):
                problems.append(f"{REQUIRED_CONFIG} must contain a JSON object")
            else:
                rank = config.get("r")
                if isinstance(rank, int) and rank > MAX_LORA_RANK:
                    problems.append(f"LoRA rank r={rank} exceeds maximum allowed rank {MAX_LORA_RANK}")

    weight_paths = [adapter_dir / filename for filename in WEIGHT_FILES if (adapter_dir / filename).is_file()]
    if not weight_paths:
        problems.append("missing adapter weight file: expected adapter_model.safetensors or adapter_model.bin")

    files: list[Path] = []
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(adapter_dir)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix not in ALLOWED_SUFFIXES:
            continue
        files.append(path)

    return files, problems


def package_adapter(adapter_dir: Path, output_zip: Path, overwrite: bool) -> int:
    resolved_adapter = resolve_path(adapter_dir)
    resolved_output = ensure_output_under_submission_zip(output_zip)
    files, problems = validate_packaging_inputs(resolved_adapter)

    if problems:
        print("Cannot package adapter:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if resolved_output.exists() and not overwrite:
        print(f"Output zip already exists; pass --overwrite to replace it: {resolved_output}", file=sys.stderr)
        return 1

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(resolved_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(resolved_adapter)))

    print(f"Packaged {len(files)} files into {resolved_output}")
    for path in files:
        print(f"  - {path.relative_to(resolved_adapter)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path, help="Directory containing real adapter files.")
    parser.add_argument("--output-zip", required=True, type=Path, help="Output zip path under submission/submission_zip/.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output zip.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return package_adapter(args.adapter_dir, args.output_zip, args.overwrite)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
