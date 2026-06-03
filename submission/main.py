#!/usr/bin/env python3
"""CLI entrypoint for lightweight submission utilities."""

from __future__ import annotations

import sys


sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from submission.scripts import (
        analyze_errors,
        build_sft_jsonl,
        check_env,
        check_model_cache,
        inspect_data,
        make_eval_subset,
        make_prediction_template,
        package_adapter,
        run_inference,
        score_outputs,
        train_lora_sft,
        validate_adapter,
        validate_submission_layout,
    )
else:
    from .scripts import (
        analyze_errors,
        build_sft_jsonl,
        check_env,
        check_model_cache,
        inspect_data,
        make_eval_subset,
        make_prediction_template,
        package_adapter,
        run_inference,
        score_outputs,
        train_lora_sft,
        validate_adapter,
        validate_submission_layout,
    )


COMMANDS = {
    "validate": validate_submission_layout.main,
    "inspect-data": inspect_data.main,
    "build-sft": build_sft_jsonl.main,
    "check-env": check_env.main,
    "check-model-cache": check_model_cache.main,
    "make-eval-subset": make_eval_subset.main,
    "make-prediction-template": make_prediction_template.main,
    "score-outputs": score_outputs.main,
    "analyze-errors": analyze_errors.main,
    "run-inference": run_inference.main,
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
    print("  check-env     Check local training/inference dependency availability")
    print("  check-model-cache  Check a local model cache without loading weights")
    print("  make-eval-subset  Build a small local proxy eval subset")
    print("  make-prediction-template  Build an empty prediction JSONL template")
    print("  score-outputs  Score predictions with the local proxy metric")
    print("  analyze-errors  Write a Markdown error analysis report")
    print("  run-inference  Run guarded local model inference")
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
