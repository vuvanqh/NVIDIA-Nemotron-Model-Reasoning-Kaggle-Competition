#!/usr/bin/env python3
"""Create an empty prediction template for a local evaluation subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_NAME = "prediction_template.jsonl"


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def choose_output_path(positional_output: Path | None, flag_output: Path | None) -> Path:
    if positional_output is not None and flag_output is not None:
        if resolve_path(positional_output) != resolve_path(flag_output):
            raise ValueError("provide output either positionally or with --output, not both")
    return flag_output or positional_output or (submission_root() / "data" / DEFAULT_OUTPUT_NAME)


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


def build_template_record(record: dict[str, Any], row_number: int, include_puzzle: bool, source_path: Path) -> dict[str, Any]:
    row_id = record.get("id")
    if row_id in (None, ""):
        raise ValueError(f"{source_path}: row {row_number}: missing id")

    template = {
        "id": row_id,
        "prediction": "",
    }
    if include_puzzle:
        template["puzzle"] = record.get("puzzle", "")
    return template


def make_prediction_template(
    input_path: Path,
    output_path: Path,
    include_puzzle: bool,
    allow_output_outside_submission_data: bool,
) -> int:
    rows = read_jsonl(input_path)
    output_rows = [
        build_template_record(record, row_number, include_puzzle, input_path)
        for row_number, record in enumerate(rows, start=1)
    ]

    resolved_output = ensure_output_under_submission_data(output_path, allow_output_outside_submission_data)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        for record in output_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Wrote prediction template: {resolved_output}")
    print("Prediction fields are intentionally empty.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Eval subset JSONL input.")
    parser.add_argument("output_path", nargs="?", type=Path, help="Output prediction template path under submission/data/.")
    parser.add_argument("--output", dest="output_flag", type=Path, help="Output prediction template path under submission/data/.")
    parser.add_argument("--include-puzzle", action="store_true", help="Include puzzle text for human-readable templates.")
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
        return make_prediction_template(
            input_path=args.input,
            output_path=output_path,
            include_puzzle=args.include_puzzle,
            allow_output_outside_submission_data=args.allow_output_outside_submission_data,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
