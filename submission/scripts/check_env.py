#!/usr/bin/env python3
"""Check import availability for submission training and inference dependencies.

This command does not load models, initialize CUDA, download files, or import
heavy ML modules. It only checks whether modules can be found in the active
Python environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path


DEFAULT_MODULES = [
    "torch",
    "transformers",
    "mamba_ssm",
    "peft",
    "datasets",
    "trl",
    "safetensors",
]


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def enable_submission_vendor_imports() -> None:
    vendor_root = submission_root() / "vendor"
    if vendor_root.is_dir():
        vendor_path = str(vendor_root)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


def package_version(module_name: str) -> str | None:
    candidates = {
        "safetensors": "safetensors",
        "transformers": "transformers",
        "datasets": "datasets",
        "torch": "torch",
        "mamba_ssm": "mamba-ssm",
        "peft": "peft",
        "trl": "trl",
    }
    distribution = candidates.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_cuda_summary() -> tuple[bool | None, str]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return None, "torch is missing"
    try:
        import torch
    except ImportError as exc:
        return None, f"torch import failed: {exc}"

    available = bool(torch.cuda.is_available())
    details = [
        f"torch version: {getattr(torch, '__version__', 'unknown')}",
        f"torch cuda build: {getattr(torch.version, 'cuda', None)}",
        f"cuda available: {available}",
        f"cuda device count: {torch.cuda.device_count() if available else 0}",
    ]
    if available:
        details.append(f"cuda device 0: {torch.cuda.get_device_name(0)}")
    return available, "; ".join(details)


def check_modules(module_names: list[str], require_cuda: bool) -> int:
    missing: list[str] = []
    print("Submission environment preflight")
    for module_name in module_names:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            missing.append(module_name)
            print(f"  - {module_name}: missing")
            continue

        version = package_version(module_name)
        suffix = f" ({version})" if version else ""
        print(f"  - {module_name}: available{suffix}")

    print("Model loading: not attempted")
    print("Network access: not attempted")
    cuda_available, cuda_details = torch_cuda_summary()
    print(f"CUDA runtime: {cuda_details}")

    if require_cuda and cuda_available is not True:
        missing.append("cuda-enabled torch runtime")
    if missing:
        print(f"Missing modules: {', '.join(missing)}")
        return 1
    print("All checked modules are available.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Module name to check. May be passed multiple times. Defaults to the submission dependency set.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless the active torch runtime can use CUDA.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    enable_submission_vendor_imports()
    return check_modules(args.modules or DEFAULT_MODULES, require_cuda=args.require_cuda)


if __name__ == "__main__":
    raise SystemExit(main())
