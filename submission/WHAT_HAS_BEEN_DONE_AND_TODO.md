# What Has Been Done And What To Do Next

This is the handoff note for the NVIDIA Nemotron reasoning challenge scaffold. It merges the previous state notes, project plan, runbooks, experiment log, and smoke reports into one file.

## Current State

`submission/` is a local scaffold for the NVIDIA Nemotron reasoning challenge. It includes data inspection, SFT sample conversion, dry-run training validation, adapter validation, adapter packaging checks, local proxy evaluation tooling, prompt formatting utilities, a guarded base-model inference entrypoint, an environment preflight command, and a model-cache preflight command.

The project is not competition-ready yet. There is no real adapter, no final package zip, no completed base-model inference output, and no official evaluation result.

Use the project venv explicitly on this Windows workspace:

```powershell
.venv\Scripts\python.exe -m submission.main <command>
```

The default `python` command is not reliable in this shell, and `py` points to Python 2.7.

## What Has Been Done

The workspace scaffold is in place:

- `submission/main.py` exposes the local utility commands.
- `submission/scripts/` contains data, scoring, validation, training dry-run, inference, environment, and cache-preflight utilities.
- `submission/configs/` contains example LoRA/SFT config files.
- `submission/adapters/` is reserved for real adapter artifacts.
- `submission/submission_zip/` is reserved for final adapter packages.
- `submission/vendor/` contains the small vendored import shim needed by the Nemotron custom code path.

Local proxy evaluation is implemented:

- `make-eval-subset` builds a small validation subset.
- `make-prediction-template` creates empty prediction templates.
- `score-outputs` scores predictions against references and supports `--overwrite`.
- `analyze-errors` writes Markdown error analysis reports.
- The scorer prioritizes the last `\boxed{...}` expression and falls back to numeric extraction only when no boxed answer is present.

Local data artifacts currently present:

- `submission/data/sft_sample.jsonl`: 10-row chat-style SFT smoke sample.
- `submission/data/sft_train_sample_100.jsonl`: 100-row chat-style SFT train sample from `data_prep/nemotron_train_v1.jsonl`.
- `submission/data/sft_val_sample_100.jsonl`: 100-row chat-style SFT validation sample from `data_prep/nemotron_val_v1.jsonl`.
- `submission/data/eval_subset_50.jsonl`: 50-row local proxy eval subset from `data_prep/nemotron_val_v1.jsonl`, sampled with seed 42.
- `submission/data/prediction_template_50.jsonl`: 50-row prediction template with intentionally empty `prediction` fields.
- `submission/data/scored_template_50.jsonl`: scored output for the empty prediction template.

The empty-template smoke score is expected and is not a model result:

- total examples: 50
- correct: 0
- incorrect: 50
- missing predictions: 50
- accuracy: 0.000000
- boxed-answer rate: 0.000000
- numeric-fallback rate: 0.000000

Inference scaffolding is implemented:

- `run-inference` is a guarded inference command.
- It refuses remote model identifiers unless `--allow-download` is passed.
- It has dry-run mode, output path checks, prompt formatting, adapter metadata checks, dtype options, offload options, and GPU/CPU memory controls.
- `submission/scripts/prompting.py` builds prompts that ask for concise reasoning and exactly one final boxed answer.
- `submission/model_dwnld.py` downloads the KaggleHub model into an external cache such as `E:\kagglehub`, not into the repo.

Preflight commands are implemented:

- `check-env` checks import availability for `torch`, `transformers`, `mamba_ssm`, `peft`, `datasets`, `trl`, and `safetensors`.
- `check-model-cache` checks a local model cache without loading weights.
- A model cache path has been used at `E:\kagglehub\models\metric\nemotron-3-nano-30b-a3b-bf16\transformers\default\1`.
- The latest model-cache preflight detected `config.json`, tokenizer files, and all 13 expected safetensors shards with no zero-byte shards.

Training and adapter packaging scaffolding exists:

- `train-lora` is dry-run-first and validates inputs.
- Dry-run training validation passes for `sft_train_sample_100.jsonl` and `sft_val_sample_100.jsonl`.
- `validate-adapter` validates adapter metadata and weight markers.
- `package-adapter` refuses to package unless a real adapter directory contains `adapter_config.json` and either `adapter_model.safetensors` or `adapter_model.bin`.

## Problems Run Into

Real base-model inference has not completed.

Earlier inference attempts reached the Nemotron custom model import and failed before writing predictions because `mamba_ssm` was missing from the active runtime.

A later smoke attempt progressed into model weight loading/offload. The captured stderr tail showed loading around 15% of 6243 weight entries, but no prediction file was written. The recorded smoke process was no longer running when checked. No `submission/data/base_predictions_smoke_1.jsonl` file exists.

The external offload folder was populated heavily during that attempt:

```text
E:\kagglehub\offload\nemotron-base
```

This suggests the run reached weight dispatch/offload but stopped before generation completed.

The project venv has had `torch`, `transformers`, and `safetensors` available. Adapter training additionally needs `peft`, `datasets`, and `trl`. The Transformers backend for this Nemotron model needs `mamba_ssm` or a compatible runtime path.

## What Is There To Do

The next major gate is a real base-model baseline. Do not start serious LoRA training before this exists.

1. Re-check the local environment:

```powershell
.venv\Scripts\python.exe -m submission.main validate
.venv\Scripts\python.exe -m submission.main check-env
.venv\Scripts\python.exe -m submission.main check-model-cache E:\kagglehub\models\metric\nemotron-3-nano-30b-a3b-bf16\transformers\default\1 --expected-shards 13
```

2. Run a one-row real inference smoke test and confirm it writes:

```text
submission/data/base_predictions_smoke_1.jsonl
```

3. Run base inference on the 50-row proxy set:

```powershell
.venv\Scripts\python.exe -m submission.main run-inference `
  --input submission/data/eval_subset_50.jsonl `
  --output submission/data/base_predictions_50.jsonl `
  --model-name-or-path E:\kagglehub\models\metric\nemotron-3-nano-30b-a3b-bf16\transformers\default\1 `
  --backend transformers `
  --max-tokens 512 `
  --temperature 0.0 `
  --top-p 1.0 `
  --trust-remote-code `
  --offload-folder E:\kagglehub\offload\nemotron-base `
  --torch-dtype float16 `
  --gpu-memory 6GiB `
  --cpu-memory 8GiB
```

4. Score and analyze the base run:

```powershell
.venv\Scripts\python.exe -m submission.main score-outputs `
  --references submission/data/eval_subset_50.jsonl `
  --predictions submission/data/base_predictions_50.jsonl `
  --output submission/data/scored_base_50.jsonl `
  --overwrite

.venv\Scripts\python.exe -m submission.main analyze-errors `
  submission/data/scored_base_50.jsonl `
  submission/reports/error_report_base_50.md
```

5. Use the base error report to decide the first training emphasis. Possible targets are final-answer formatting, arithmetic, unit conversion, physics formulas, numeral conversion, symbolic reasoning, task interpretation, or another dominant failure mode.

6. Implement real LoRA SFT training only after the base baseline exists. The first serious adapter should be rank 16 unless the evidence points elsewhere.

7. Train a real adapter under:

```text
submission/adapters/nemotron_lora_r16_v1/
```

Expected adapter files:

```text
adapter_config.json
adapter_model.safetensors
```

`adapter_model.bin` is acceptable if the runtime emits that format.

8. Validate and package the adapter:

```powershell
.venv\Scripts\python.exe -m submission.main validate-adapter submission/adapters/nemotron_lora_r16_v1
.venv\Scripts\python.exe -m submission.main package-adapter `
  --adapter-dir submission/adapters/nemotron_lora_r16_v1 `
  --output-zip submission/submission_zip/submission.zip
```

9. Evaluate the adapter against the same 50-row proxy set, compare against base metrics, and only move to rank 32 if rank 16 appears capacity-limited.

10. Prepare the final notebook/write-up after a real adapter, local metrics, and package exist.

## Experiment Tracking Template

Every result should point to saved files. Do not fill metrics from memory.

| Run ID | Adapter | Rank | Data | LR | Epochs | Eval Set | Accuracy | Boxed Rate | Numeric Fallback Rate | Files | Notes |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---|---|
| base_50 | none | 0 | none | - | - | eval_subset_50 | TBD | TBD | TBD | `submission/data/base_predictions_50.jsonl`, `submission/data/scored_base_50.jsonl`, `submission/reports/error_report_base_50.md` | Waiting for real inference. |
| r16_v1 | LoRA | 16 | TBD after baseline errors | TBD | TBD | eval_subset_50 | TBD | TBD | TBD | `submission/adapters/nemotron_lora_r16_v1/`, `submission/data/scored_adapter_50.jsonl` | Do not train until `base_50` exists. |
| r32_v1 | LoRA | 32 | TBD | TBD | TBD | eval_subset_50 | TBD | TBD | TBD | `submission/adapters/nemotron_lora_r32_v1/` | Use only if rank 16 appears capacity-limited. |

## Key Rules

- Do not train before measuring the base model.
- Do not create fake adapter artifacts.
- Do not place full base model weights in `submission/`.
- Do not claim official performance from the local proxy metric.
- Do not trust synthetic data unless it is independently verified.
- Keep final answers in exactly one final `\boxed{...}` expression.
- Validate adapter layout before packaging.
- Keep generated large files out of git.

## Definition Of Done

The project is done when a real LoRA adapter exists, the adapter rank is at most 32, the adapter validates locally, the final zip package is created, the package matches the official competition requirements, local proxy score and error analysis are recorded, a submission has been made, and a public notebook/write-up describes the method and results.
