#!/usr/bin/env python3
"""Run guarded local model inference for eval subsets.

Dry-run mode validates inputs and prints the planned inference configuration
without importing heavy ML libraries, downloading models, or writing fake
predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .prompting import build_reasoning_prompt


DEFAULT_BACKEND = "transformers"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
TORCH_DTYPE_CHOICES = ("auto", "float16", "bfloat16", "float32")


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def enable_submission_vendor_imports() -> None:
    vendor_root = submission_root() / "vendor"
    if vendor_root.is_dir():
        vendor_path = str(vendor_root)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


def force_optional_cuda_backend_fallbacks() -> None:
    try:
        from transformers.utils import import_utils
    except ImportError:
        return

    # The local vendor shim only provides gated RMSNorm for the model's slow
    # PyTorch path; it is not the full mamba-ssm CUDA backend.
    import_utils.is_mamba_2_ssm_available = lambda: False


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
        raise ValueError(f"--output must be under {data_root}; pass --allow-output-outside-submission-data to override") from None
    return output_path


def looks_like_local_path(model_name_or_path: str) -> bool:
    model_path = Path(model_name_or_path).expanduser()
    return model_path.exists() or model_name_or_path.startswith((".", "/", "~"))


def validate_model_access(model_name_or_path: str, allow_download: bool, dry_run: bool) -> None:
    if not model_name_or_path:
        raise ValueError("--model-name-or-path is required")

    is_local_like = looks_like_local_path(model_name_or_path)
    if not is_local_like and not allow_download:
        raise ValueError(
            "model name is not a local path; pass --allow-download to permit remote model loading explicitly"
        )

    if not dry_run and is_local_like and not Path(model_name_or_path).expanduser().exists():
        raise ValueError(f"local model path does not exist: {model_name_or_path}")


def validate_adapter_dir(adapter_dir: Path | None) -> Path | None:
    if adapter_dir is None:
        return None

    resolved_adapter = resolve_path(adapter_dir)
    if not resolved_adapter.is_dir():
        raise ValueError(f"adapter directory not found: {resolved_adapter}")
    if not (resolved_adapter / "adapter_config.json").is_file():
        raise ValueError(f"adapter directory is missing adapter_config.json: {resolved_adapter}")
    if not any((resolved_adapter / filename).is_file() for filename in ADAPTER_WEIGHT_FILES):
        expected = " or ".join(ADAPTER_WEIGHT_FILES)
        raise ValueError(f"adapter directory is missing adapter weights ({expected}): {resolved_adapter}")
    return resolved_adapter


def read_eval_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if not path.is_file():
        raise ValueError(f"input JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: row {row_number}: invalid JSON: {exc}") from None
            if not isinstance(record, dict):
                raise ValueError(f"{path}: row {row_number}: top-level JSON value is not an object")
            if record.get("id") in (None, ""):
                raise ValueError(f"{path}: row {row_number}: missing id")
            if record.get("puzzle") in (None, ""):
                raise ValueError(f"{path}: row {row_number}: missing puzzle")
            rows.append(record)
    return rows


def prediction_record(source: dict[str, Any], prediction: str, include_puzzle: bool) -> dict[str, Any]:
    record = {
        "id": source["id"],
        "prediction": prediction,
    }
    if source.get("task_type") not in (None, ""):
        record["task_type"] = source.get("task_type")
    if include_puzzle:
        record["puzzle"] = source.get("puzzle", "")
    return record


def write_predictions(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_existing_prediction_ids(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()

    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{output_path}: row {row_number}: invalid JSON in existing predictions: {exc}") from None
            if not isinstance(record, dict):
                raise ValueError(f"{output_path}: row {row_number}: existing prediction is not an object")
            row_id = record.get("id")
            if row_id not in (None, ""):
                ids.add(str(row_id))
    return ids


def append_prediction(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def generate_with_transformers(
    rows: list[dict[str, Any]],
    model_name_or_path: str,
    adapter_dir: Path | None,
    offload_folder: Path | None,
    torch_dtype_name: str,
    gpu_memory: str | None,
    cpu_memory: str | None,
    use_cache: bool,
    use_mamba_kernels: bool | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    allow_download: bool,
    trust_remote_code: bool,
    include_puzzle: bool,
    prediction_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    enable_submission_vendor_imports()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "missing transformers backend dependencies; install torch and transformers in the target runtime"
        ) from exc
    force_optional_cuda_backend_fallbacks()

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=not allow_download,
    )
    config.use_cache = use_cache
    if use_mamba_kernels is not None and hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = use_mamba_kernels

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=not allow_download,
    )
    dtype_by_name = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": not allow_download,
        "device_map": "auto",
        "torch_dtype": dtype_by_name[torch_dtype_name],
        "low_cpu_mem_usage": True,
        "config": config,
    }
    max_memory: dict[int | str, str] = {}
    if gpu_memory:
        max_memory[0] = gpu_memory
    if cpu_memory:
        max_memory["cpu"] = cpu_memory
    if max_memory:
        model_kwargs["max_memory"] = max_memory
    if offload_folder is not None:
        offload_folder.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_folder)
        model_kwargs["offload_state_dict"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("--adapter-dir requires peft to be installed in the target runtime") from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True)

    model.eval()
    generated: list[dict[str, Any]] = []
    do_sample = temperature > 0.0

    for source in rows:
        prompt = build_reasoning_prompt(str(source["puzzle"]))
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": use_cache,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_tokens = inputs["input_ids"].shape[-1]
        continuation_ids = output_ids[0][prompt_tokens:]
        prediction = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
        record = prediction_record(source, prediction, include_puzzle=include_puzzle)
        generated.append(record)
        if prediction_callback is not None:
            prediction_callback(record)

    return generated


def print_dry_run_plan(
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    adapter_dir: Path | None,
) -> None:
    first_prompt = build_reasoning_prompt(str(rows[0]["puzzle"])) if rows else ""
    preview = first_prompt[:800]
    if len(first_prompt) > len(preview):
        preview += "..."

    plan = {
        "mode": "dry-run",
        "backend": args.backend,
        "input": str(input_path),
        "output": str(output_path),
        "rows_planned": len(rows),
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "offload_folder": str(resolve_path(args.offload_folder)) if args.offload_folder else None,
        "torch_dtype": args.torch_dtype,
        "gpu_memory": args.gpu_memory,
        "cpu_memory": args.cpu_memory,
        "use_cache": args.use_cache,
        "use_mamba_kernels": args.use_mamba_kernels,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "allow_download": args.allow_download,
        "trust_remote_code": args.trust_remote_code,
        "first_id": rows[0].get("id") if rows else None,
        "first_prompt_preview": preview,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("Dry run complete. No ML libraries were imported, no model was loaded, and no predictions were written.")


def run_vllm_backend(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    raise RuntimeError("vllm backend is a placeholder scaffold; use --backend transformers for the implemented path")


def run(args: argparse.Namespace) -> int:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in the range (0, 1]")

    input_path = resolve_path(args.input)
    output_path = ensure_output_under_submission_data(args.output, args.allow_output_outside_submission_data)
    validate_model_access(args.model_name_or_path, args.allow_download, args.dry_run)
    adapter_dir = validate_adapter_dir(args.adapter_dir)
    offload_folder = resolve_path(args.offload_folder) if args.offload_folder else None
    rows = read_eval_jsonl(input_path, args.limit)

    if args.dry_run:
        print_dry_run_plan(rows, input_path, output_path, args, adapter_dir)
        return 0

    if output_path.exists() and not args.overwrite and not args.resume:
        raise ValueError(f"output file already exists; pass --overwrite to replace it: {output_path}")

    existing_ids: set[str] = set()
    output_mode = "w"
    if args.resume:
        existing_ids = read_existing_prediction_ids(output_path)
        rows_before_resume = len(rows)
        rows = [row for row in rows if str(row["id"]) not in existing_ids]
        skipped = rows_before_resume - len(rows)
        output_mode = "a"
        print(f"Resume mode: found {len(existing_ids)} existing prediction ids; skipped {skipped} input rows.")

    output_handle = None
    streamed_count = 0

    def stream_prediction(record: dict[str, Any]) -> None:
        nonlocal output_handle, streamed_count
        if output_handle is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open(output_mode, encoding="utf-8")
        append_prediction(output_handle, record)
        streamed_count += 1

    if not rows:
        if args.resume and not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
        print(f"No rows to run. Existing predictions are in {output_path}")
        return 0

    if args.backend == "transformers":
        try:
            predictions = generate_with_transformers(
                rows=rows,
                model_name_or_path=args.model_name_or_path,
                adapter_dir=adapter_dir,
                offload_folder=offload_folder,
                torch_dtype_name=args.torch_dtype,
                gpu_memory=args.gpu_memory,
                cpu_memory=args.cpu_memory,
                use_cache=args.use_cache,
                use_mamba_kernels=args.use_mamba_kernels,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                allow_download=args.allow_download,
                trust_remote_code=args.trust_remote_code,
                include_puzzle=args.include_puzzle,
                prediction_callback=stream_prediction,
            )
        finally:
            if output_handle is not None:
                output_handle.close()
    elif args.backend == "vllm":
        predictions = run_vllm_backend()
    else:
        raise ValueError(f"unsupported backend: {args.backend}")

    if streamed_count == 0:
        write_predictions(output_path, predictions)

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    print("These are local model predictions, not official competition results.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Eval JSONL containing id and puzzle fields.")
    parser.add_argument("--output", required=True, type=Path, help="Prediction JSONL output path under submission/data/.")
    parser.add_argument("--model-name-or-path", required=True, help="Local model path, or remote identifier with --allow-download.")
    parser.add_argument("--adapter-dir", type=Path, help="Optional LoRA adapter directory with adapter config and weights.")
    parser.add_argument("--offload-folder", type=Path, help="Optional disk offload folder for device_map=auto model loading.")
    parser.add_argument("--torch-dtype", choices=TORCH_DTYPE_CHOICES, default="auto", help="Model load dtype. Defaults to auto.")
    parser.add_argument("--gpu-memory", help="Max memory for GPU 0, e.g. 6GiB. Passed to from_pretrained max_memory.")
    parser.add_argument("--cpu-memory", help="Max CPU memory, e.g. 8GiB. Passed to from_pretrained max_memory.")
    parser.add_argument("--use-cache", action="store_true", help="Enable generation KV/cache state. Disabled by default for lower-memory local smoke runs.")
    parser.add_argument(
        "--use-mamba-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request the model's fast Mamba kernels when available. Defaults to false for the local fallback path.",
    )
    parser.add_argument("--backend", choices=("transformers", "vllm"), default=DEFAULT_BACKEND, help="Inference backend.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature. Defaults to 0.0.")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Nucleus sampling top-p. Defaults to 1.0.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum input rows to process.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the inference plan without loading a model or writing predictions.")
    parser.add_argument("--allow-download", action="store_true", help="Permit remote model loading explicitly. Disabled by default.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to model/tokenizer loading.")
    parser.add_argument("--include-puzzle", action="store_true", help="Include puzzle text in prediction output rows.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file in real inference mode.")
    parser.add_argument("--resume", action="store_true", help="Append missing predictions and skip ids already present in --output.")
    parser.add_argument(
        "--allow-output-outside-submission-data",
        action="store_true",
        help="Allow --output outside submission/data/. Use only for explicit local experiments.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (RuntimeError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
