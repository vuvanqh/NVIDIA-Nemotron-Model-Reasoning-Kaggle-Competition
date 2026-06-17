#!/usr/bin/env python3
"""Validate a LoRA adapter directory using only local metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAX_LORA_RANK = 32
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def read_adapter_config(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    problems: list[str] = []
    if not path.is_file():
        return None, [f"missing required file: {path.name}"]

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON in {path.name}: {exc}"]

    if not isinstance(data, dict):
        return None, [f"{path.name} must contain a JSON object"]
    return data, problems


def extract_lora_rank(config: dict[str, Any]) -> int | None:
    value = config.get("r")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def validate_adapter(adapter_dir: Path) -> int:
    if not adapter_dir.is_dir():
        print(f"Adapter directory not found: {adapter_dir}")
        print("Adapter validation failed.")
        return 1

    config, problems = read_adapter_config(adapter_dir / "adapter_config.json")
    warnings: list[str] = []
    rank: int | None = None
    if config is not None:
        rank = extract_lora_rank(config)
        if rank is None:
            warnings.append("LoRA rank field `r` was not found or is not numeric.")
        elif rank > MAX_LORA_RANK:
            problems.append(f"LoRA rank r={rank} exceeds maximum allowed rank {MAX_LORA_RANK}.")

    weight_paths = [adapter_dir / filename for filename in WEIGHT_FILES if (adapter_dir / filename).is_file()]
    if not weight_paths:
        warnings.append("No adapter weight file found: expected adapter_model.safetensors or adapter_model.bin.")

    print(f"Adapter directory: {adapter_dir}")
    if rank is not None:
        print(f"LoRA rank: {rank}")
    if weight_paths:
        print("Adapter weight files:")
        for path in weight_paths:
            print(f"  - {path.name}")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if problems:
        print("Failures:")
        for problem in problems:
            print(f"  - {problem}")
        print("Adapter validation failed.")
        return 1

    print("Adapter validation passed.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path, help="Path to a LoRA adapter directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return validate_adapter(args.adapter_dir)


if __name__ == "__main__":
    raise SystemExit(main())
