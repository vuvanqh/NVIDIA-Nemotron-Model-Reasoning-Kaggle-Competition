#!/usr/bin/env python3
"""CLI entrypoint for lightweight submission utilities."""

from __future__ import annotations

import sys


sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from submission.scripts import (
        build_sft_jsonl,
        inspect_data,
        package_adapter,
        train_lora_sft,
        validate_adapter,
        validate_submission_layout,
    )
else:
    from .scripts import (
        build_sft_jsonl,
        inspect_data,
        package_adapter,
        train_lora_sft,
        validate_adapter,
        validate_submission_layout,
    )


COMMANDS = {
    "validate": validate_submission_layout.main,
    "inspect-data": inspect_data.main,
    "build-sft": build_sft_jsonl.main,
    "train-lora": train_lora_sft.main,
    "validate-adapter": validate_adapter.main,
    "package-adapter": package_adapter.main,
}


def print_help() -> None:
    print("Usage: python -m submission.main <command> [options]")
    print()
    print("Commands:")
    print("  validate      Validate local submission workspace layout")
    print("  inspect-data  Inspect a prepared JSONL data file")
    print("  build-sft     Build SFT-style JSONL under submission/data/")
    print("  train-lora    Validate inputs and dry-run future LoRA SFT training")
    print("  validate-adapter  Validate local LoRA adapter metadata")
    print("  package-adapter   Package a real adapter into submission_zip/")
    print()
    print("Run a command with --help for command-specific options.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return validate_submission_layout.main([])
    if args[0] in {"-h", "--help"}:
        print_help()
        return 0

    command = args.pop(0)
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_help()
        return 2

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
