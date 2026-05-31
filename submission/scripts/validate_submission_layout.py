#!/usr/bin/env python3
"""Validate the lightweight submission folder layout.

This script checks local files and directories only. It does not download models,
load adapters, or run the competition benchmark.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


REQUIRED_DIRS = [
    "adapters",
    "configs",
    "data",
    "notebooks",
    "scripts",
    "submission_zip",
]

REQUIRED_FILES = [
    "README.md",
    "REPO_STATE.md",
    "requirements.txt",
    "__init__.py",
    "configs/lora_config.example.yaml",
    "configs/sft_lora_config.example.yaml",
    "scripts/__init__.py",
    "scripts/build_sft_jsonl.py",
    "scripts/inspect_data.py",
    "scripts/package_adapter.py",
    "scripts/train_lora_sft.py",
    "scripts/validate_adapter.py",
    "scripts/validate_submission_layout.py",
    "data/README.md",
    "adapters/README.md",
    "submission_zip/README.md",
    "notebooks/README.md",
]

ADAPTER_MARKERS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
]

TOOL_FILES = {
    "data inspector": "scripts/inspect_data.py",
    "SFT builder": "scripts/build_sft_jsonl.py",
    "training scaffold": "scripts/train_lora_sft.py",
    "adapter validator": "scripts/validate_adapter.py",
    "package builder": "scripts/package_adapter.py",
}


def find_training_data(root: Path) -> list[Path]:
    data_root = root / "data"
    return sorted(path for path in data_root.rglob("*.jsonl") if path.is_file())


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_paths(paths: list[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)) for path in paths]


def find_adapter_markers(root: Path) -> list[Path]:
    found: list[Path] = []
    adapter_root = root / "adapters"
    for marker in ADAPTER_MARKERS:
        found.extend(adapter_root.rglob(marker))
    return sorted(found)


def find_real_adapter_dirs(root: Path) -> list[Path]:
    adapter_root = root / "adapters"
    found: list[Path] = []
    for config_path in adapter_root.rglob("adapter_config.json"):
        adapter_dir = config_path.parent
        if any((adapter_dir / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")):
            found.append(adapter_dir)
    return sorted(set(found))


def find_packages(root: Path) -> list[Path]:
    package_root = root / "submission_zip"
    return sorted(package_root.glob("*.zip"))


def package_has_adapter_files(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, FileNotFoundError):
        return False
    return "adapter_config.json" in names and bool(
        {"adapter_model.safetensors", "adapter_model.bin"} & names
    )


def read_adapter_rank(adapter_dir: Path) -> int | None:
    config_path = adapter_dir / "adapter_config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    value = config.get("r") if isinstance(config, dict) else None
    return value if isinstance(value, int) else None


def workspace_state(data_files: list[Path], real_adapter_dirs: list[Path], packages: list[Path]) -> str:
    if packages and real_adapter_dirs:
        return "final-zip-present"
    if packages:
        return "final-zip-present-without-adapter"
    if real_adapter_dirs:
        return "adapter-present"
    if data_files:
        return "training-data-prepared"
    return "scaffold-only"


def print_paths(label: str, paths: list[Path], root: Path) -> None:
    if paths:
        print(f"{label}:")
        for path in relative_paths(paths, root):
            print(f"  - {path}")
    else:
        print(f"{label}: none")


def print_tool_status(root: Path) -> None:
    print("Tool scaffolding:")
    for label, relative_path in TOOL_FILES.items():
        status = "present" if (root / relative_path).is_file() else "missing"
        print(f"  - {label}: {status}")


def validate(strict_data: bool, strict_adapter: bool, strict_package: bool) -> int:
    root = submission_root()
    missing_dirs = [path for path in REQUIRED_DIRS if not (root / path).is_dir()]
    missing_files = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    data_files = find_training_data(root)
    adapter_markers = find_adapter_markers(root)
    real_adapter_dirs = find_real_adapter_dirs(root)
    packages = find_packages(root)
    valid_package_candidates = [path for path in packages if package_has_adapter_files(path)]
    state = workspace_state(data_files, real_adapter_dirs, packages)

    print(f"Submission root: {root}")
    print(f"Workspace state: {state}")

    if missing_dirs:
        print("Missing directories:")
        for path in missing_dirs:
            print(f"  - {path}")

    if missing_files:
        print("Missing files:")
        for path in missing_files:
            print(f"  - {path}")

    print_tool_status(root)
    print_paths("Training data JSONL files under data/", data_files, root)
    print_paths("Adapter markers found", adapter_markers, root)
    print_paths("Real adapter directories", real_adapter_dirs, root)
    print_paths("Package files found", packages, root)
    print_paths("Package files containing adapter config and weights", valid_package_candidates, root)

    for adapter_dir in real_adapter_dirs:
        rank = read_adapter_rank(adapter_dir)
        if rank is not None:
            print(f"Adapter rank for {adapter_dir.relative_to(root)}: {rank}")

    errors = bool(missing_dirs or missing_files)
    if strict_data and not data_files:
        print("Strict data check failed: no JSONL training data is present under data/.")
        errors = True
    if strict_adapter and not real_adapter_dirs:
        print("Strict adapter check failed: no complete adapter directory with config and weights is present.")
        errors = True
    if strict_package and not packages:
        print("Strict package check failed: no .zip package file is present.")
        errors = True

    if errors:
        print("Layout validation failed.")
        return 1

    if not real_adapter_dirs:
        print("Competition readiness: not ready; no real adapter marker files are present.")
    elif not packages:
        print("Competition readiness: adapter files present, but no final .zip package is present.")
    elif not valid_package_candidates:
        print("Competition readiness: zip files exist, but none contain both adapter_config.json and adapter weights.")
    else:
        print("Competition readiness: package files contain adapter markers, but official format is not validated here.")

    print("Layout validation passed.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Fail when no JSONL training data is present under data/.",
    )
    parser.add_argument(
        "--strict-adapter",
        action="store_true",
        help="Fail when no adapter marker files are present under adapters/.",
    )
    parser.add_argument(
        "--strict-package",
        action="store_true",
        help="Fail when no .zip package file is present under submission_zip/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return validate(
        strict_data=args.require_data,
        strict_adapter=args.strict_adapter,
        strict_package=args.strict_package,
    )


if __name__ == "__main__":
    raise SystemExit(main())
