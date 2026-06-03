# NVIDIA Nemotron Reasoning Submission Workspace

This folder contains the submission-side workflow for the NVIDIA Nemotron reasoning challenge. It is currently a scaffold with local proxy evaluation tools, SFT sample conversion, adapter/package validation, dependency preflights, and a guarded base-model inference entrypoint.

Read `submission/WHAT_HAS_BEEN_DONE_AND_TODO.md` for the current handoff state, completed work, known blockers, and next steps.

## Validate The Scaffold

Run from the repository root:

```bash
python3 -m submission.main validate
```

On this Windows workspace, use the venv Python explicitly:

```bash
.venv\Scripts\python.exe -m submission.main validate
```

Normal validation checks local scaffold files and reports current adapter/package absence as expected scaffold state. It does not prove competition readiness.

Strict checks are available when you intentionally want failures until real artifacts exist:

```bash
python3 -m submission.main validate --require-data --strict-adapter --strict-package
```

Check dependency availability without loading a model or touching the network:

```bash
.venv\Scripts\python.exe -m submission.main check-env
```

For the local Transformers backend, `mamba_ssm` must be available because the Nemotron H custom model code imports it.

Check an external model cache without loading weights:

```bash
.venv\Scripts\python.exe -m submission.main check-model-cache E:\kagglehub\models\metric\nemotron-3-nano-30b-a3b-bf16\transformers\default\1 --expected-shards 13
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
  --trust-remote-code \
  --dry-run
```

The prompt asks for concise reasoning and requires the final answer in exactly one `\boxed{...}` expression. Intermediate values should not be boxed.

## Run Real Base-Model Inference

Run real inference only in a runtime with the model already available locally, suitable GPU memory, and required dependencies installed. The default mode refuses remote model identifiers unless `--allow-download` is explicitly passed.

To download the KaggleHub model locally first, use the submission downloader. It defaults the KaggleHub cache to `E:\kagglehub` so the full model is stored on the larger `E:` drive:

```bash
python3 submission/model_dwnld.py
```

Override the cache location only if another large local drive is available:

```bash
python3 submission/model_dwnld.py --cache-dir E:\kagglehub
```

Before real inference, the model cache preflight should pass.

```bash
python3 -m submission.main run-inference \
  --input submission/data/eval_subset_50.jsonl \
  --output submission/data/base_predictions_50.jsonl \
  --model-name-or-path /path/to/local/Nemotron-3-Nano-30B \
  --backend transformers \
  --max-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0 \
  --trust-remote-code \
  --offload-folder E:\kagglehub\offload\nemotron-base \
  --torch-dtype float16 \
  --gpu-memory 6GiB \
  --cpu-memory 8GiB
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

Pass `--overwrite` when replacing an existing scored output file.

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

Build the current 100-row dry-run samples:

```bash
python3 -m submission.main build-sft data_prep/nemotron_train_v1.jsonl submission/data/sft_train_sample_100.jsonl --limit 100
python3 -m submission.main build-sft data_prep/nemotron_val_v1.jsonl submission/data/sft_val_sample_100.jsonl --limit 100
```

Run a dry-run LoRA plan check:

```bash
python3 -m submission.main train-lora --dry-run --train-file submission/data/sft_sample.jsonl --eval-file submission/data/sft_sample.jsonl --output-dir submission/adapters/dry_run_adapter
```

Run the current 100-row dry-run check:

```bash
python3 -m submission.main train-lora --dry-run --train-file submission/data/sft_train_sample_100.jsonl --eval-file submission/data/sft_val_sample_100.jsonl --output-dir submission/adapters/dry_run_adapter
```

Real training remains intentionally unimplemented in this scaffold. The command validates inputs and prints a plan in dry-run mode only.

After a real adapter is trained elsewhere, validate and package it with:

```bash
python3 -m submission.main validate-adapter submission/adapters/nemotron_lora
python3 -m submission.main package-adapter --adapter-dir submission/adapters/nemotron_lora --output-zip submission/submission_zip/nemotron_lora_adapter.zip
```

The package command refuses to create a zip unless the adapter directory contains `adapter_config.json` and an adapter weight file such as `adapter_model.safetensors` or `adapter_model.bin`.

See `submission/WHAT_HAS_BEEN_DONE_AND_TODO.md` for the saved workflow notes and next gates.

## Boundaries

- Do not place full base model weights in `submission/`.
- Do not treat the local proxy score as an official benchmark result.
- Do not claim competition readiness until a real adapter exists, the official package format is confirmed, and the official evaluation path is run.
