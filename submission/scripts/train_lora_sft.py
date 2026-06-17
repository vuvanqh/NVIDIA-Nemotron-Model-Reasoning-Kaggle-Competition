#!/usr/bin/env python3
"""Dry-run-first scaffold for future Nemotron LoRA SFT training.

The default path is validation and planning only. Real training requires running
without --dry-run and a local/runtime model path, plus installed ML dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .build_sft_jsonl import find_boxed_spans


DEFAULT_MODEL_NAME = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
DEFAULT_OUTPUT_DIR = Path("submission/adapters/nemotron_lora")


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_output_under_adapters(path: Path) -> Path:
    output_path = resolve_path(path)
    adapter_root = (submission_root() / "adapters").resolve()
    try:
        output_path.relative_to(adapter_root)
    except ValueError:
        raise ValueError(f"--output-dir must be under {adapter_root}") from None
    return output_path


def load_jsonl_examples(path: Path, limit: int) -> tuple[list[dict[str, Any]], int, list[str]]:
    examples: list[dict[str, Any]] = []
    errors: list[str] = []
    total_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            if len(examples) >= limit:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("top-level JSON value is not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"row {row_number}: {exc}")
                continue
            examples.append(record)
    return examples, total_rows, errors


def assistant_content(record: dict[str, Any]) -> str | None:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                content = message.get("content")
                return content if isinstance(content, str) else None
    output = record.get("output")
    return output if isinstance(output, str) else None


def user_content(record: dict[str, Any]) -> str | None:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                content = message.get("content")
                return content if isinstance(content, str) else None
    puzzle = record.get("puzzle")
    return puzzle if isinstance(puzzle, str) else None


def validate_sft_examples(path: Path, sample_limit: int) -> tuple[int, list[str], list[dict[str, Any]]]:
    examples, total_rows, parse_errors = load_jsonl_examples(path, sample_limit)
    problems = list(parse_errors)
    preview: list[dict[str, Any]] = []

    for index, record in enumerate(examples, start=1):
        user = user_content(record)
        assistant = assistant_content(record)
        if not user:
            problems.append(f"sample {index}: missing user content")
        if not assistant:
            problems.append(f"sample {index}: missing assistant content")
            continue

        boxed = find_boxed_spans(assistant)
        if len(boxed) != 1 or boxed[0][1] != len(assistant.rstrip()):
            problems.append(f"sample {index}: assistant response must end with exactly one boxed answer")

        preview.append(
            {
                "id": record.get("id"),
                "user_chars": len(user or ""),
                "assistant_chars": len(assistant),
                "boxed_answer": boxed[0][2] if len(boxed) == 1 else None,
            }
        )

    return total_rows, problems, preview


def import_training_dependencies() -> dict[str, Any]:
    missing: list[str] = []
    imports: dict[str, Any] = {}
    for module_name in ("torch", "datasets", "transformers", "peft", "trl"):
        try:
            imports[module_name] = __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing training dependencies: {joined}")
    return imports


def looks_like_local_path(model_name_or_path: str) -> bool:
    model_path = Path(model_name_or_path).expanduser()
    return model_path.exists() or model_name_or_path.startswith((".", "/", "~"))


def run_real_training(args: argparse.Namespace, output_dir: Path) -> int:
    if not args.model_name_or_path:
        print("Real training requires --model-name-or-path.", file=sys.stderr)
        return 1
    if not looks_like_local_path(args.model_name_or_path) and not args.allow_remote_model_download:
        print(
            "Refusing to start training with a non-local model identifier. "
            "Use a local/runtime model path, or pass --allow-remote-model-download knowingly.",
            file=sys.stderr,
        )
        return 1

    try:
        import_training_dependencies()
    except RuntimeError as exc:
        print(f"{exc}. Install dependencies in the target training environment first.", file=sys.stderr)
        return 1

    print("Training dependencies are importable.")
    print(f"Planned adapter output directory: {output_dir}")
    print("Real training loop is not implemented in this scaffold yet.")
    return 1


def print_plan(args: argparse.Namespace, output_dir: Path, train_rows: int, eval_rows: int, preview: list[dict[str, Any]]) -> None:
    plan = {
        "mode": "dry-run" if args.dry_run else "train",
        "train_file": str(resolve_path(args.train_file)),
        "eval_file": str(resolve_path(args.eval_file)) if args.eval_file else None,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "output_dir": str(output_dir),
        "model_name_or_path": args.model_name_or_path,
        "config": str(resolve_path(args.config)) if args.config else None,
        "sample_limit": args.sample_limit,
        "allow_remote_model_download": args.allow_remote_model_download,
        "sample_preview": preview,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def run(args: argparse.Namespace) -> int:
    train_file = resolve_path(args.train_file)
    eval_file = resolve_path(args.eval_file) if args.eval_file else train_file
    output_dir = ensure_output_under_adapters(args.output_dir)

    missing_files = [path for path in (train_file, eval_file) if not path.is_file()]
    if missing_files:
        for path in missing_files:
            print(f"Input JSONL file not found: {path}", file=sys.stderr)
        return 1

    if args.config:
        config_path = resolve_path(args.config)
        if not config_path.is_file():
            print(f"Config file not found: {config_path}", file=sys.stderr)
            return 1

    train_rows, train_problems, preview = validate_sft_examples(train_file, args.sample_limit)
    eval_rows, eval_problems, _eval_preview = validate_sft_examples(eval_file, args.sample_limit)
    problems = [f"train: {problem}" for problem in train_problems]
    problems.extend(f"eval: {problem}" for problem in eval_problems)

    if problems:
        print("SFT data validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print_plan(args, output_dir, train_rows, eval_rows, preview)

    if args.dry_run:
        print("Dry run complete. No model was loaded, downloaded, or trained.")
        return 0

    return run_real_training(args, output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path, help="SFT training JSONL file.")
    parser.add_argument("--eval-file", type=Path, help="SFT evaluation JSONL file. Defaults to train file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Adapter output directory under submission/adapters/.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME, help="Local/runtime model path or explicit model identifier.")
    parser.add_argument("--config", type=Path, help="Optional training config file.")
    parser.add_argument("--sample-limit", type=int, default=5, help="Examples to inspect before training. Defaults to 5.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the planned training config without training.")
    parser.add_argument(
        "--allow-remote-model-download",
        action="store_true",
        help="Allow a non-local model identifier when real training is explicitly requested.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
