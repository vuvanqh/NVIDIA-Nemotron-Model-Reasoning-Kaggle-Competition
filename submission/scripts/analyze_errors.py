#!/usr/bin/env python3
"""Generate a Markdown error report from local proxy scored JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ERROR_TYPES = (
    "missing prediction",
    "no boxed answer",
    "numeric fallback used",
    "exact mismatch",
    "numeric mismatch",
)


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def ensure_output_under_reports(path: Path, allow_outside: bool) -> Path:
    output_path = resolve_path(path)
    if allow_outside:
        return output_path

    reports_root = (submission_root() / "reports").resolve()
    try:
        output_path.relative_to(reports_root)
    except ValueError:
        raise ValueError(f"output path must be under {reports_root}; pass --allow-output-outside-submission-reports to override") from None
    if output_path.suffix.lower() != ".md":
        raise ValueError("output report must end with .md")
    return output_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Scored JSONL file not found: {path}")

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


def task_type_key(record: dict[str, Any]) -> str:
    task_type = record.get("task_type")
    if task_type in (None, ""):
        return "unknown"
    return str(task_type)


def row_error_types(record: dict[str, Any]) -> list[str]:
    if record.get("correct") is True:
        return []

    errors: list[str] = []
    if record.get("missing_prediction") is True:
        errors.append("missing prediction")
    if record.get("has_boxed_answer") is not True:
        errors.append("no boxed answer")
    if record.get("used_numeric_fallback") is True:
        errors.append("numeric fallback used")

    match_type = record.get("match_type")
    if match_type == "numeric":
        errors.append("numeric mismatch")
    elif match_type != "missing":
        errors.append("exact mismatch")

    return errors


def summarize(rows: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, Counter[str]]]:
    overall: Counter[str] = Counter()
    by_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_task_error_types: dict[str, Counter[str]] = defaultdict(Counter)

    for record in rows:
        task_type = task_type_key(record)
        correct = record.get("correct") is True
        overall["total"] += 1
        by_task_counts[task_type]["total"] += 1
        if correct:
            overall["correct"] += 1
            by_task_counts[task_type]["correct"] += 1
        else:
            overall["incorrect"] += 1
            by_task_counts[task_type]["incorrect"] += 1
            for error_type in row_error_types(record):
                overall[error_type] += 1
                by_task_error_types[task_type][error_type] += 1

    return overall, by_task_counts, by_task_error_types


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def make_markdown(
    rows: list[dict[str, Any]],
    scored_path: Path,
    overall: Counter[str],
    by_task_counts: dict[str, Counter[str]],
    by_task_error_types: dict[str, Counter[str]],
) -> str:
    lines: list[str] = []
    lines.append("# Local Proxy Error Report")
    lines.append("")
    lines.append("This report is based on the local proxy metric only. It is not the official competition metric.")
    lines.append("")
    lines.append(f"- Scored input: `{scored_path}`")
    lines.append(f"- Total examples: {overall['total']}")
    lines.append(f"- Correct: {overall['correct']}")
    lines.append(f"- Incorrect: {overall['incorrect']}")
    lines.append(f"- Accuracy: {format_rate(overall['correct'], overall['total'])}")
    lines.append("")
    lines.append("## Error Type Counts")
    lines.append("")
    lines.append("Counts are non-exclusive; one incorrect row can contribute to multiple categories.")
    lines.append("")
    lines.append("| Error type | Count |")
    lines.append("| --- | ---: |")
    for error_type in ERROR_TYPES:
        lines.append(f"| {error_type} | {overall[error_type]} |")
    lines.append("")
    lines.append("## Errors By Task Type")
    lines.append("")
    lines.append("| Task type | Examples | Correct | Incorrect | Accuracy | Missing prediction | No boxed answer | Numeric fallback used | Exact mismatch | Numeric mismatch |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for task_type in sorted(by_task_counts):
        counts = by_task_counts[task_type]
        errors = by_task_error_types[task_type]
        lines.append(
            "| "
            + " | ".join(
                [
                    task_type,
                    str(counts["total"]),
                    str(counts["correct"]),
                    str(counts["incorrect"]),
                    format_rate(counts["correct"], counts["total"]),
                    str(errors["missing prediction"]),
                    str(errors["no boxed answer"]),
                    str(errors["numeric fallback used"]),
                    str(errors["exact mismatch"]),
                    str(errors["numeric mismatch"]),
                ]
            )
            + " |"
        )
    lines.append("")

    incorrect_examples = [record for record in rows if record.get("correct") is not True][:10]
    lines.append("## First Incorrect Examples")
    lines.append("")
    if not incorrect_examples:
        lines.append("No incorrect examples found.")
    else:
        for record in incorrect_examples:
            lines.append(f"- `{record.get('id')}` `{task_type_key(record)}`: {record.get('primary_error_type')} | target=`{record.get('target_answer')}` | extracted=`{record.get('extracted_answer')}`")
    lines.append("")
    return "\n".join(lines)


def analyze_errors(scored_path: Path, output_path: Path, allow_output_outside_reports: bool) -> int:
    rows = read_jsonl(scored_path)
    resolved_output = ensure_output_under_reports(output_path, allow_output_outside_reports)
    overall, by_task_counts, by_task_error_types = summarize(rows)
    report = make_markdown(rows, scored_path, overall, by_task_counts, by_task_error_types)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(report, encoding="utf-8")

    print(f"Read scored rows: {overall['total']}")
    print(f"Incorrect rows: {overall['incorrect']}")
    print(f"Wrote error report: {resolved_output}")
    print("Official metric: not evaluated by this script")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored_jsonl", type=Path, help="Scored JSONL output from score-outputs.")
    parser.add_argument("output", type=Path, help="Markdown report path under submission/reports/.")
    parser.add_argument(
        "--allow-output-outside-submission-reports",
        action="store_true",
        help="Allow output outside submission/reports/. Use only for explicit local experiments.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return analyze_errors(
            scored_path=args.scored_jsonl,
            output_path=args.output,
            allow_output_outside_reports=args.allow_output_outside_submission_reports,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
