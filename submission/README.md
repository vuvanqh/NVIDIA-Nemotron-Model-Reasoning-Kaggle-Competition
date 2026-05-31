# NVIDIA Nemotron Reasoning Submission Workspace

This folder is the submission-side workspace for a lightweight Nemotron LoRA reasoning-challenge workflow. The root repository and `data_prep/` files are treated as read-only context from here.

The `../data_prep/` path is relative to this README. The command examples below assume they are run from the repository root.

## Current scope

- Keep challenge-specific submission files under `submission/`.
- Use prepared JSONL data from `../data_prep/` as read-only inputs unless copied intentionally into `submission/data/`.
- Train or place LoRA adapter outputs under `submission/adapters/`.
- Build any final package artifacts under `submission/submission_zip/`.
- Validate layout with a local check before packaging.

## Observed input data

The prepared data outside this folder currently uses JSONL records with these fields:

- `id`
- `task_type`
- `puzzle`
- `target_answer`
- `output`
- `nemotron_tokens`

The observed split files are:

- `../data_prep/nemotron_train_v1.jsonl`
- `../data_prep/nemotron_val_v1.jsonl`

Those files were inspected as read-only context and were not modified by this setup pass.

## Suggested workflow

1. Review `configs/lora_config.example.yaml` and copy it to a run-specific config when training details are known.
2. Keep local or Kaggle-mounted data references lightweight. Do not commit large datasets or model weights here.
3. Inspect the prepared read-only validation data:

   ```bash
   python3 -m submission.main inspect-data data_prep/nemotron_val_v1.jsonl --limit 3
   ```

4. Build a small SFT sample under `submission/data/`:

   ```bash
   python3 -m submission.main build-sft data_prep/nemotron_val_v1.jsonl submission/data/sft_sample.jsonl --limit 10
   ```

5. Validate the current submission workspace layout:

   ```bash
   python3 -m submission.main validate
   ```

6. Run a dry-run training plan check:

   ```bash
   python3 -m submission.main train-lora --dry-run --train-file submission/data/sft_sample.jsonl --eval-file submission/data/sft_sample.jsonl --output-dir submission/adapters/dry_run_adapter
   ```

7. Train a LoRA adapter in the target runtime once assumptions are confirmed, then write real adapter artifacts under `adapters/`.
8. Validate the real adapter directory:

   ```bash
   python3 -m submission.main validate-adapter submission/adapters/nemotron_lora
   ```

9. Package final challenge artifacts under `submission_zip/` once the required competition format is confirmed.

## Utility commands

Inspect a JSONL file without loading it fully into memory:

```bash
python3 -m submission.main inspect-data data_prep/nemotron_train_v1.jsonl --limit 5
```

Preview SFT conversion without writing a file:

```bash
python3 -m submission.main build-sft data_prep/nemotron_val_v1.jsonl submission/data/sft_sample.jsonl --limit 2 --dry-run
```

Build a small sample file:

```bash
python3 -m submission.main build-sft data_prep/nemotron_val_v1.jsonl submission/data/sft_sample.jsonl --limit 10
```

Validate scaffold, data, adapter, and zip-package state:

```bash
python3 -m submission.main validate
```

Run a future LoRA SFT training dry-run:

```bash
python3 -m submission.main train-lora --dry-run --train-file submission/data/sft_sample.jsonl --eval-file submission/data/sft_sample.jsonl --output-dir submission/adapters/dry_run_adapter
```

Validate a real adapter directory after training:

```bash
python3 -m submission.main validate-adapter submission/adapters/nemotron_lora
```

Package a real adapter after validation:

```bash
python3 -m submission.main package-adapter --adapter-dir submission/adapters/nemotron_lora --output-zip submission/submission_zip/nemotron_lora_adapter.zip
```

The package command refuses to create a zip unless the adapter directory contains `adapter_config.json` and an adapter weight file such as `adapter_model.safetensors` or `adapter_model.bin`.

## Notes

- This scaffold does not download the Nemotron base model.
- This scaffold does not run the competition benchmark.
- The SFT sample builder does not train a model.
- The training command is dry-run-first. Real training is not implemented or run in this setup pass.
- Real adapter outputs should be written under `submission/adapters/`.
- A final package must ultimately contain a compatible LoRA adapter for Nemotron-3-Nano-30B, but this workspace does not prove leaderboard or benchmark compatibility.
- The validator only checks expected local files and folders; it does not prove competition compatibility.
