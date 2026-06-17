#!/usr/bin/env python3
"""Build a small local evaluation subset from prepared reference JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


DEFAULT_LIMIT = 50
PRESERVED_FIELDS = ("id", "task_type", "puzzle", "target_answer")
REQUIRED_FIELDS = ("id", "puzzle", "target_answer")


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def default_output_path() -> Path:
    return submission_root() / "data" / "eval_subset.jsonl"


def choose_output_path(positional_output: Path | None, flag_output: Path | None) -> Path:
    if positional_output is not None and flag_output is not None:
        if resolve_path(positional_output) != resolve_path(flag_output):
            raise ValueError("provide output either positionally or with --output, not both")
    return flag_output or positional_output or default_output_path()


def ensure_output_under_submission_data(path: Path, allow_outside: bool) -> Path:
    output_path = resolve_path(path)
    if allow_outside:
        return output_path

    data_root = (submission_root() / "data").resolve()
    try:
        output_path.relative_to(data_root)
    except ValueError:
        raise ValueError(f"output path must be under {data_root}; pass --allow-output-outside-submission-data to override") from None
    return output_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Input JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: row {row_number}: invalid JSON: {exc}") from None
            if not isinstance(record, dict):
                raise ValueError(f"{path}: row {row_number}: top-level JSON value is not an object")
            rows.append(record)
    return rows


def make_subset_record(record: dict[str, Any], row_number: int, source_path: Path) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if record.get(field) in (None, "")]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source_path}: row {row_number}: missing required fields: {joined}")

    return {field: record.get(field) for field in PRESERVED_FIELDS}


def build_subset(
    input_path: Path,
    output_path: Path,
    limit: int,
    seed: int | None,
    task_types: list[str] | None,
    allow_output_outside_submission_data: bool,
) -> int:
    if limit <= 0:
        raise ValueError("--limit must be positive")

    source_rows = read_jsonl(input_path)
    selected: list[dict[str, Any]] = []
    allowed_task_types = set(task_types or [])

    for row_number, record in enumerate(source_rows, start=1):
        if allowed_task_types and str(record.get("task_type")) not in allowed_task_types:
            continue
        selected.append(make_subset_record(record, row_number, input_path))

    if seed is not None:
        random.Random(seed).shuffle(selected)

    selected = selected[:limit]
    if not selected:
        raise ValueError("no rows matched the requested filters")

    resolved_output = ensure_output_under_submission_data(output_path, allow_output_outside_submission_data)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Input rows: {len(source_rows)}")
    print(f"Rows after filters: {len(selected)}")
    print(f"Wrote eval subset: {resolved_output}")
    print("Note: this is a local proxy evaluation subset, not an official benchmark split.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Reference JSONL input, for example data_prep/nemotron_val_v1.jsonl.")
    parser.add_argument("output_path", nargs="?", type=Path, help="Output JSONL path under submission/data/.")
    parser.add_argument("--output", dest="output_flag", type=Path, help="Output JSONL path under submission/data/.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Maximum rows to write. Defaults to {DEFAULT_LIMIT}.")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle deterministically with this seed before slicing.")
    parser.add_argument(
        "--task-type",
        action="append",
        default=None,
        help="Only include this task_type. May be passed multiple times.",
    )
    parser.add_argument(
        "--allow-output-outside-submission-data",
        action="store_true",
        help="Allow output outside submission/data/. Use only for explicit local experiments.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_path = choose_output_path(args.output_path, args.output_flag)
        return build_subset(
            input_path=args.input,
            output_path=output_path,
            limit=args.limit,
            seed=args.seed,
            task_types=args.task_type,
            allow_output_outside_submission_data=args.allow_output_outside_submission_data,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
