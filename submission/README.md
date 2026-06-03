# NVIDIA Nemotron Reasoning Submission Workspace

This folder contains the submission-side workflow for the NVIDIA Nemotron reasoning challenge. It is currently a scaffold with local proxy evaluation tools and a guarded base-model inference entrypoint. Files outside `submission/`, including `../data_prep/`, are treated as read-only context.

## Current Position

- Prepared challenge data exists outside this folder under `../data_prep/`.
- `submission/data/sft_sample.jsonl` contains a 10-row SFT-format smoke-test sample.
- `submission/data/eval_subset_50.jsonl` contains a small local proxy evaluation subset.
- Local proxy evaluation tools exist for subset creation, prediction templating, scoring, and error analysis.
- A guarded `run-inference` command exists for base-model inference in a GPU/runtime environment.
- No real model inference has been run unless a real prediction file such as `submission/data/base_predictions_50.jsonl` exists.
- No LoRA training has been run by this scaffold.
- No real adapter exists under `submission/adapters/`.
- No final zip package exists under `submission/submission_zip/`.
- The local proxy metric is not the official competition metric.

## Validate The Scaffold

Run from the repository root:

```bash
python3 -m submission.main validate
```

Normal validation checks local scaffold files and reports current adapter/package absence as expected scaffold state. It does not prove competition readiness.

Strict checks are available when you intentionally want failures until real artifacts exist:

```bash
python3 -m submission.main validate --require-data --strict-adapter --strict-package
```

## Build A Local Eval Subset

Create a small subset under `submission/data/`:

```bash
python3 -m submission.main make-eval-subset data_prep/nemotron_val_v1.jsonl submission/data/eval_subset_50.jsonl --limit 50 --seed 42
```

The subset builder preserves only `id`, `task_type`, `puzzle`, and `target_answer`. It uses a small default limit so it does not copy the full dataset by default.

## Dry-Run Base-Model Inference

Dry-run mode validates the input file, output path, adapter metadata if provided, model access policy, and prompt formatting. It does not import heavy ML libraries, load a model, download a model, or write predictions.

```bash
python3 -m submission.main run-inference \
  --input submission/data/eval_subset_50.jsonl \
  --output submission/data/base_predictions_50.jsonl \
  --model-name-or-path /path/to/local/Nemotron-3-Nano-30B \
  --backend transformers \
  --limit 3 \
  --dry-run
```

The prompt asks for concise reasoning and requires the final answer in exactly one `\boxed{...}` expression. Intermediate values should not be boxed.

## Run Real Base-Model Inference

Run real inference only in a runtime with the model already available locally, suitable GPU memory, and required dependencies installed. The default mode refuses remote model identifiers unless `--allow-download` is explicitly passed.

```bash
python3 -m submission.main run-inference \
  --input submission/data/eval_subset_50.jsonl \
  --output submission/data/base_predictions_50.jsonl \
  --model-name-or-path /path/to/local/Nemotron-3-Nano-30B \
  --backend transformers \
  --max-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0
```

For an adapter-backed run after a real adapter exists:

```bash
python3 -m submission.main run-inference \
  --input submission/data/eval_subset_50.jsonl \
  --output submission/data/adapter_predictions_50.jsonl \
  --model-name-or-path /path/to/local/Nemotron-3-Nano-30B \
  --adapter-dir submission/adapters/nemotron_lora \
  --backend transformers
```

Do not start LoRA training before establishing a base-model inference baseline and scoring it locally.

## Create A Prediction Template

Create an empty prediction file matching the eval subset ids:

```bash
python3 -m submission.main make-prediction-template submission/data/eval_subset_50.jsonl submission/data/prediction_template_50.jsonl
```

The prediction schema is:

```json
{"id": "example-id", "prediction": ""}
```

For manual review, include puzzle text:

```bash
python3 -m submission.main make-prediction-template submission/data/eval_subset_50.jsonl submission/data/prediction_template_50_with_puzzles.jsonl --include-puzzle
```

## Score Predictions With The Local Proxy Metric

Score a prediction JSONL against a reference subset:

```bash
python3 -m submission.main score-outputs --references submission/data/eval_subset_50.jsonl --predictions submission/data/base_predictions_50.jsonl --output submission/data/scored_base_50.jsonl
```

The scorer:

- matches rows by `id`;
- compares `prediction` against `target_answer`;
- prioritizes the last `\boxed{...}` answer in a prediction;
- falls back to numeric extraction only when no boxed answer exists;
- supports normalized exact string match;
- supports relative numeric tolerance;
- writes a scored JSONL report;
- prints total examples, correct, incorrect, missing predictions, accuracy, boxed-answer rate, and numeric-fallback rate.

This is a local proxy metric only. It is not the official leaderboard or competition metric.

## Generate An Error Report

```bash
python3 -m submission.main analyze-errors submission/data/scored_base_50.jsonl submission/reports/error_report_base_50.md
```

The report groups incorrect rows by `task_type` when available and counts non-exclusive error indicators:

- missing prediction;
- no boxed answer;
- numeric fallback used;
- exact mismatch;
- numeric mismatch.

## Existing Empty-Template Smoke Check

The current empty prediction template can be scored as a smoke check:

```bash
python3 -m submission.main score-outputs --references submission/data/eval_subset_50.jsonl --predictions submission/data/prediction_template_50.jsonl --output submission/data/scored_template_50.jsonl
python3 -m submission.main analyze-errors submission/data/scored_template_50.jsonl submission/reports/error_report_template_50.md
```

It should score 0 accuracy because the predictions are intentionally empty.

## SFT And Adapter Utilities

Build a small SFT sample:

```bash
python3 -m submission.main build-sft data_prep/nemotron_val_v1.jsonl submission/data/sft_sample.jsonl --limit 10 --overwrite
```

Run a dry-run LoRA plan check:

```bash
python3 -m submission.main train-lora --dry-run --train-file submission/data/sft_sample.jsonl --eval-file submission/data/sft_sample.jsonl --output-dir submission/adapters/dry_run_adapter
```

Real training remains intentionally unimplemented in this scaffold. The command validates inputs and prints a plan in dry-run mode only.

After a real adapter is trained elsewhere, validate and package it with:

```bash
python3 -m submission.main validate-adapter submission/adapters/nemotron_lora
python3 -m submission.main package-adapter --adapter-dir submission/adapters/nemotron_lora --output-zip submission/submission_zip/nemotron_lora_adapter.zip
```

The package command refuses to create a zip unless the adapter directory contains `adapter_config.json` and an adapter weight file such as `adapter_model.safetensors` or `adapter_model.bin`.

## Boundaries

- Do not place full base model weights in `submission/`.
- Do not treat the local proxy score as an official benchmark result.
- Do not claim competition readiness until a real adapter exists, the official package format is confirmed, and the official evaluation path is run.
