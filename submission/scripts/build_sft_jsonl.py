#!/usr/bin/env python3
"""Build a small SFT-style JSONL file from prepared Nemotron data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BOXED_PREFIX = "\\boxed"


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_output_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve(strict=False)


def ensure_output_under_submission_data(path: Path) -> Path:
    output_path = resolve_output_path(path)
    data_root = (submission_root() / "data").resolve()
    try:
        output_path.relative_to(data_root)
    except ValueError:
        raise ValueError(f"Output path must be under {data_root}") from None
    return output_path


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


def remove_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end, _content in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces).rstrip()


def normalize_answer(value: Any) -> str:
    return str(value).strip()


def build_assistant_response(record: dict[str, Any], assistant_field: str) -> str:
    source_text = str(record.get(assistant_field) or record.get("output") or "").strip()
    target_answer = normalize_answer(record.get("target_answer", ""))
    spans = find_boxed_spans(source_text)
    boxed_answer = target_answer or (spans[-1][2].strip() if spans else "")

    if not boxed_answer:
        raise ValueError("record has no target_answer and no boxed answer in output")

    stripped_source = source_text.rstrip()
    if len(spans) == 1 and spans[0][1] == len(stripped_source):
        existing_answer = spans[0][2].strip()
        if not target_answer or existing_answer == target_answer:
            return stripped_source

    source_without_boxes = remove_spans(source_text, spans)
    final_box = f"{BOXED_PREFIX}{{{boxed_answer}}}"
    if source_without_boxes:
        return f"{source_without_boxes}\n\n{final_box}"
    return final_box


def make_sft_record(record: dict[str, Any], row_number: int, assistant_field: str) -> dict[str, Any]:
    puzzle = str(record.get("puzzle") or "").strip()
    if not puzzle:
        raise ValueError("record is missing puzzle text")

    assistant_response = build_assistant_response(record, assistant_field=assistant_field)
    boxed_spans = find_boxed_spans(assistant_response)
    if len(boxed_spans) != 1 or boxed_spans[0][1] != len(assistant_response.rstrip()):
        raise ValueError("assistant response does not end with exactly one boxed answer")

    metadata = {
        "source_row": row_number,
        "id": record.get("id"),
        "task_type": record.get("task_type"),
        "target_answer": record.get("target_answer"),
        "nemotron_tokens": record.get("nemotron_tokens"),
    }

    return {
        "id": record.get("id"),
        "messages": [
            {"role": "user", "content": puzzle},
            {"role": "assistant", "content": assistant_response},
        ],
        "metadata": metadata,
    }


def build_sft_jsonl(
    input_path: Path,
    output_path: Path,
    limit: int | None,
    dry_run: bool,
    overwrite: bool,
    assistant_field: str,
) -> int:
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")
    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    resolved_output = ensure_output_under_submission_data(output_path)
    if resolved_output.exists() and not overwrite and not dry_run:
        print(f"Output file already exists; pass --overwrite to replace it: {resolved_output}", file=sys.stderr)
        return 1

    written = 0
    skipped = 0

    output_handle = None
    try:
        if not dry_run:
            resolved_output.parent.mkdir(parents=True, exist_ok=True)
            output_handle = resolved_output.open("w", encoding="utf-8")

        with input_path.open("r", encoding="utf-8") as input_handle:
            for row_number, line in enumerate(input_handle, start=1):
                if limit is not None and written >= limit:
                    break
                if not line.strip():
                    continue

                try:
                    source_record = json.loads(line)
                    if not isinstance(source_record, dict):
                        raise ValueError("top-level JSON value is not an object")
                    sft_record = make_sft_record(
                        source_record,
                        row_number=row_number,
                        assistant_field=assistant_field,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    skipped += 1
                    print(f"Skipping row {row_number}: {exc}", file=sys.stderr)
                    continue

                line_out = json.dumps(sft_record, ensure_ascii=False)
                if dry_run:
                    print(line_out)
                else:
                    assert output_handle is not None
                    output_handle.write(f"{line_out}\n")
                written += 1
    finally:
        if output_handle is not None:
            output_handle.close()

    action = "Would write" if dry_run else "Wrote"
    destination = "stdout" if dry_run else str(resolved_output)
    print(f"{action} {written} examples to {destination}", file=sys.stderr)
    if skipped:
        print(f"Skipped {skipped} malformed or incomplete rows", file=sys.stderr)

    return 0 if written > 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prepared input JSONL file.")
    parser.add_argument("output", type=Path, help="Output JSONL path under submission/data/.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum examples to write.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated JSONL instead of writing.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--assistant-field",
        choices=("output", "target_answer"),
        default="output",
        help="Source field for assistant responses. Defaults to output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return build_sft_jsonl(
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            assistant_field=args.assistant_field,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
