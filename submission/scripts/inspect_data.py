#!/usr/bin/env python3
"""Inspect prepared Nemotron JSONL data without loading it all into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_FIELDS = (
    "id",
    "task_type",
    "puzzle",
    "target_answer",
    "output",
    "nemotron_tokens",
)


def preview(value: Any, max_chars: int = 160) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def numeric_token_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def summarize_sample(row_number: int, record: dict[str, Any]) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "row": row_number,
        "fields": sorted(record),
    }
    for field in ("id", "task_type", "target_answer", "nemotron_tokens"):
        if field in record:
            sample[field] = record[field]
    if "puzzle" in record:
        sample["puzzle_preview"] = preview(record["puzzle"])
    if "output" in record:
        sample["output_preview"] = preview(record["output"])
    return sample


def inspect_jsonl(path: Path, limit: int, required_fields: tuple[str, ...]) -> int:
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    if not path.is_file():
        print(f"Input file not found: {path}")
        return 1

    total_rows = 0
    malformed_rows: list[tuple[int, str]] = []
    detected_fields: set[str] = set()
    missing_required: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    token_count = 0
    token_min: int | None = None
    token_max: int | None = None
    token_sum = 0
    token_non_numeric = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_rows.append((line_number, str(exc)))
                continue
            if not isinstance(record, dict):
                malformed_rows.append((line_number, "top-level JSON value is not an object"))
                continue

            detected_fields.update(record)
            for field in required_fields:
                if field not in record or record[field] in (None, ""):
                    missing_required[field] += 1

            task_type = record.get("task_type")
            if task_type not in (None, ""):
                task_type_counts[str(task_type)] += 1

            if "nemotron_tokens" in record:
                token_value = numeric_token_value(record.get("nemotron_tokens"))
                if token_value is None:
                    token_non_numeric += 1
                else:
                    token_count += 1
                    token_sum += token_value
                    token_min = token_value if token_min is None else min(token_min, token_value)
                    token_max = token_value if token_max is None else max(token_max, token_value)

            if len(samples) < limit:
                samples.append(summarize_sample(line_number, record))

    print(f"File: {path}")
    print(f"Total rows: {total_rows}")
    print("Detected fields:")
    for field in sorted(detected_fields):
        print(f"  - {field}")

    if missing_required:
        print("Missing required fields:")
        for field, count in sorted(missing_required.items()):
            print(f"  - {field}: {count} rows")
    else:
        print("Missing required fields: none")

    if task_type_counts:
        print("Rows by task_type:")
        for task_type, count in task_type_counts.most_common():
            print(f"  - {task_type}: {count}")
    else:
        print("Rows by task_type: task_type field not present or empty")

    if token_count:
        mean_tokens = token_sum / token_count
        print("nemotron_tokens statistics:")
        print(f"  - numeric rows: {token_count}")
        print(f"  - min: {token_min}")
        print(f"  - max: {token_max}")
        print(f"  - mean: {mean_tokens:.2f}")
        if token_non_numeric:
            print(f"  - non-numeric rows: {token_non_numeric}")
    else:
        print("nemotron_tokens statistics: no numeric token values found")

    if malformed_rows:
        print("Malformed rows:")
        for row_number, message in malformed_rows[:10]:
            print(f"  - row {row_number}: {message}")
        if len(malformed_rows) > 10:
            print(f"  - ... {len(malformed_rows) - 10} additional malformed rows")

    if samples:
        print(f"First {len(samples)} examples:")
        print(json.dumps(samples, ensure_ascii=False, indent=2))
    else:
        print("First examples: none")

    return 1 if malformed_rows else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a JSONL file to inspect.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of leading examples to summarize. Defaults to 5.",
    )
    parser.add_argument(
        "--required-fields",
        nargs="+",
        default=list(DEFAULT_REQUIRED_FIELDS),
        help="Fields to check for presence. Defaults to the observed split schema.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return inspect_jsonl(
        path=args.path,
        limit=args.limit,
        required_fields=tuple(args.required_fields),
    )


if __name__ == "__main__":
    raise SystemExit(main())
