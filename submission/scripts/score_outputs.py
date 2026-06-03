#!/usr/bin/env python3
"""Score prediction JSONL files with a local proxy metric.

This is not the official competition metric. It is a lightweight local proxy
for validating answer extraction, exact matching, and numeric tolerance logic.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


BOXED_PREFIX = "\\boxed"
NUMBER_PATTERN = re.compile(r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?")
LEADING_ZERO_INTEGER_PATTERN = re.compile(r"[-+]?0\d+")
DEFAULT_RELATIVE_TOLERANCE = 1e-6


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def ensure_output_under_submission_data(path: Path, allow_outside: bool) -> Path:
    output_path = resolve_path(path)
    if allow_outside:
        return output_path

    data_root = (submission_root() / "data").resolve()
    try:
        output_path.relative_to(data_root)
    except ValueError:
        raise ValueError(f"--output must be under {data_root}") from None
    return output_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL file not found: {path}")

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


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def find_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    search_at = 0
    while True:
        start = text.find(BOXED_PREFIX, search_at)
        if start == -1:
            return spans

        open_brace = start + len(BOXED_PREFIX)
        while open_brace < len(text) and text[open_brace].isspace():
            open_brace += 1

        if open_brace >= len(text) or text[open_brace] != "{":
            search_at = start + len(BOXED_PREFIX)
            continue

        depth = 0
        for index in range(open_brace, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1, text[open_brace + 1 : index]))
                    search_at = index + 1
                    break
        else:
            return spans


def parse_numeric_answer(value: Any) -> float | None:
    text = normalize_text(value).replace(",", "")
    if not text:
        return None
    if LEADING_ZERO_INTEGER_PATTERN.fullmatch(text):
        return None
    if NUMBER_PATTERN.fullmatch(text) is None:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def extract_last_number(text: str) -> str | None:
    matches = list(NUMBER_PATTERN.finditer(text.replace(",", "")))
    if not matches:
        return None
    return matches[-1].group(0)


def extract_prediction_answer(prediction: str) -> tuple[str, bool, bool]:
    spans = find_boxed_spans(prediction)
    if spans:
        return spans[-1][2].strip(), True, False

    numeric_answer = extract_last_number(prediction)
    if numeric_answer is not None:
        return numeric_answer.strip(), False, True

    return prediction.strip(), False, False


def within_relative_tolerance(predicted: float, target: float, tolerance: float) -> bool:
    scale = max(1.0, abs(target))
    return abs(predicted - target) <= tolerance * scale


def prediction_index(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row_number, record in enumerate(rows, start=1):
        row_id = record.get("id")
        if row_id in (None, ""):
            print(f"Skipping prediction row {row_number}: missing id", file=sys.stderr)
            continue
        key = str(row_id)
        if key in indexed:
            duplicates += 1
        indexed[key] = record
    return indexed, duplicates


def score_record(
    reference: dict[str, Any],
    prediction: dict[str, Any] | None,
    relative_tolerance: float,
) -> dict[str, Any]:
    row_id = str(reference.get("id"))
    target_answer = reference.get("target_answer", "")
    raw_prediction = "" if prediction is None else str(prediction.get("prediction", ""))
    missing_prediction = prediction is None or not raw_prediction.strip()

    extracted_answer, has_boxed_answer, used_numeric_fallback = extract_prediction_answer(raw_prediction)
    normalized_target = normalize_text(target_answer)
    normalized_extracted = normalize_text(extracted_answer)

    predicted_number = parse_numeric_answer(extracted_answer)
    target_number = parse_numeric_answer(target_answer)

    correct = False
    match_type = "missing" if missing_prediction else "exact"
    numeric_abs_error: float | None = None
    numeric_relative_error: float | None = None

    if not missing_prediction and normalized_extracted == normalized_target:
        correct = True
        match_type = "exact"
    elif not missing_prediction and predicted_number is not None and target_number is not None:
        match_type = "numeric"
        numeric_abs_error = abs(predicted_number - target_number)
        denominator = max(1.0, abs(target_number))
        numeric_relative_error = numeric_abs_error / denominator
        correct = within_relative_tolerance(predicted_number, target_number, relative_tolerance)

    if missing_prediction:
        primary_error_type = "missing prediction"
    elif correct:
        primary_error_type = "none"
    elif match_type == "numeric":
        primary_error_type = "numeric mismatch"
    else:
        primary_error_type = "exact mismatch"

    return {
        "id": row_id,
        "task_type": reference.get("task_type"),
        "target_answer": target_answer,
        "prediction": raw_prediction,
        "extracted_answer": extracted_answer,
        "has_boxed_answer": has_boxed_answer,
        "used_numeric_fallback": used_numeric_fallback,
        "missing_prediction": missing_prediction,
        "match_type": match_type,
        "correct": correct,
        "primary_error_type": primary_error_type,
        "numeric_abs_error": numeric_abs_error,
        "numeric_relative_error": numeric_relative_error,
    }


def score_outputs(
    references_path: Path,
    predictions_path: Path,
    output_path: Path,
    relative_tolerance: float,
    allow_output_outside_submission_data: bool,
) -> int:
    if relative_tolerance < 0:
        raise ValueError("--relative-tolerance must be non-negative")

    references = read_jsonl(references_path)
    predictions, duplicate_predictions = prediction_index(read_jsonl(predictions_path))
    resolved_output = ensure_output_under_submission_data(output_path, allow_output_outside_submission_data)

    seen_reference_ids: set[str] = set()
    scored_rows: list[dict[str, Any]] = []
    for row_number, reference in enumerate(references, start=1):
        if reference.get("id") in (None, ""):
            raise ValueError(f"{references_path}: row {row_number}: missing id")
        if reference.get("target_answer") in (None, ""):
            raise ValueError(f"{references_path}: row {row_number}: missing target_answer")

        row_id = str(reference["id"])
        if row_id in seen_reference_ids:
            raise ValueError(f"{references_path}: duplicate reference id: {row_id}")
        seen_reference_ids.add(row_id)

        scored_rows.append(
            score_record(
                reference=reference,
                prediction=predictions.get(row_id),
                relative_tolerance=relative_tolerance,
            )
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(scored_rows)
    correct = sum(1 for row in scored_rows if row["correct"])
    missing = sum(1 for row in scored_rows if row["missing_prediction"])
    boxed = sum(1 for row in scored_rows if row["has_boxed_answer"])
    numeric_fallback = sum(1 for row in scored_rows if row["used_numeric_fallback"])
    accuracy = correct / total if total else 0.0
    boxed_rate = boxed / total if total else 0.0
    numeric_fallback_rate = numeric_fallback / total if total else 0.0

    if duplicate_predictions:
        print(f"Warning: {duplicate_predictions} duplicate prediction ids were overwritten by later rows", file=sys.stderr)

    print("Local proxy metric results")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Missing predictions: {missing}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Boxed-answer rate: {boxed_rate:.6f}")
    print(f"Numeric-fallback rate: {numeric_fallback_rate:.6f}")
    print(f"Scored report: {resolved_output}")
    print("Official metric: not evaluated by this script")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path, help="Reference JSONL with id and target_answer fields.")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction JSONL with id and prediction fields.")
    parser.add_argument("--output", required=True, type=Path, help="Output scored JSONL path under submission/data/.")
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_TOLERANCE,
        help=f"Relative numeric tolerance. Defaults to {DEFAULT_RELATIVE_TOLERANCE}.",
    )
    parser.add_argument(
        "--allow-output-outside-submission-data",
        action="store_true",
        help="Allow --output outside submission/data/. Use only for explicit local experiments.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return score_outputs(
            references_path=args.references,
            predictions_path=args.predictions,
            output_path=args.output,
            relative_tolerance=args.relative_tolerance,
            allow_output_outside_submission_data=args.allow_output_outside_submission_data,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
