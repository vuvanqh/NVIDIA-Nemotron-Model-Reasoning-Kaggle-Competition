# Submission Repo State

This file summarizes the current state of `submission/`. It is not a changelog.

## Current Scope

`submission/` contains a local scaffold for the NVIDIA Nemotron reasoning challenge. It includes data inspection, SFT sample conversion, dry-run training validation, adapter validation, adapter packaging checks, local proxy evaluation tooling, prompt formatting utilities, and a guarded base-model inference entrypoint.

## Data Inside Submission

- `submission/data/sft_sample.jsonl` exists as a 10-row chat-style SFT smoke-test sample.
- `submission/data/eval_subset_50.jsonl` exists as a 50-row local proxy evaluation subset sampled from `data_prep/nemotron_val_v1.jsonl` with seed 42.
- `submission/data/prediction_template_50.jsonl` exists as a 50-row prediction template with intentionally empty `prediction` fields.
- `submission/data/scored_template_50.jsonl` exists as the scored output for the empty prediction template.
- `submission/data/base_predictions_50.jsonl` does not exist.
- `submission/data/scored_base_50.jsonl` does not exist.
- No full training or validation dataset is copied into `submission/data/`.

## Local Proxy Evaluation

The local proxy evaluation pipeline is available through these commands:

- `make-eval-subset`
- `make-prediction-template`
- `score-outputs`
- `analyze-errors`

The current smoke-check score is for an empty prediction template:

- total examples: 50
- correct: 0
- incorrect: 50
- missing predictions: 50
- accuracy: 0.000000
- boxed-answer rate: 0.000000
- numeric-fallback rate: 0.000000

This score is expected for an empty template. It is not a model result and not the official competition metric.

## Inference State

- `run-inference` exists as a guarded inference command.
- `submission/scripts/prompting.py` contains the prompt builder used by inference.
- A local Nemotron model cache directory was found at `/Users/michalkozicki/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`.
- Real inference was attempted with backend `transformers`, `temperature=0.0`, `top_p=1.0`, and `max_tokens=512`.
- The attempt stopped before model loading because the active `python3` runtime is missing backend dependencies: `torch` and `transformers`.
- The project `.venv` has `transformers` but is also missing `torch`.
- No model download was requested or allowed.
- No real base-model inference has completed in this scaffold.
- No base-model prediction file exists.

## Reports

- `submission/reports/error_report_template_50.md` exists as a Markdown error report for the empty-template smoke check.
- `submission/reports/error_report_base_50.md` does not exist because no base-model prediction file has been generated or scored.

## Base Proxy Metrics

Base-model local proxy metrics are not available because real base-model inference did not complete.

- base accuracy: not available
- boxed-answer rate: not available
- numeric-fallback rate: not available
- top base error categories: not available

## Training

- No LoRA training has been run by this scaffold.
- `train-lora` is dry-run-first and real training remains unimplemented.
- LoRA training should wait until a base-model inference baseline exists and has been scored locally.

## Adapter And Package State

- `submission/adapters/` contains no real adapter artifacts.
- No `adapter_config.json` exists under `submission/adapters/`.
- No `adapter_model.safetensors` or `adapter_model.bin` exists under `submission/adapters/`.
- `submission/submission_zip/` contains no final `.zip` package.
- The submission is not competition-ready.

## Validation State

Normal scaffold validation passes with:

```bash
python3 -m submission.main validate
```

Strict adapter and package validation is expected to fail until real adapter files and a final package are created.
