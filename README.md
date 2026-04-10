# MorphoPulse Open

Minimal open-source release for:

- foundation model pre-training
- foundation model linear-probing evaluation
- synthetic PPG generation training, evaluation, and sampling

This repository intentionally contains only the source files that are on the execution path of the released entrypoints.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for origin and license notes about the included code paths.

The repository-level license is MIT. Some files in `src/generation/` are adapted from TimeVQVAE and retain upstream attribution notes as documented in `THIRD_PARTY_NOTICES.md`.

## Repository layout

- `src/foundation/`: foundation model training and evaluation code
- `src/generation/`: synthetic generation code
- `configs/`: released configs
- `scripts/`: runnable entry scripts
- `datasets/`: placeholder directory structure only
- `docs/`: dataset and usage notes

## Environment variables

- `PROJECT_ROOT`: repository root
- `DATA_ROOT`: dataset root, defaults to `PROJECT_ROOT/datasets`
- `CHECKPOINT_ROOT`: checkpoint root, defaults to `PROJECT_ROOT/artifacts/checkpoints`
- `OUTPUT_ROOT`: output root, defaults to `PROJECT_ROOT/artifacts/outputs`

## Entrypoints

### Foundation training

```bash
bash scripts/train_foundation.sh
```

### Foundation evaluation

Set `MODEL` and `TIME` to the checkpoint filename and subdirectory you want to evaluate.

```bash
MODEL=your_model.pt TIME=your_run bash scripts/eval_foundation.sh
```

### Generation train/eval/sample

```bash
DATASET_NAME=Mesa_ppg_svri bash scripts/run_generation.sh
```

## Data policy

No real datasets are included. This repository only provides placeholder directories and README files documenting the expected structure.

## Checkpoints

No pretrained checkpoints are included. See [docs/checkpoints.md](docs/checkpoints.md) for the expected layout.

# Acknowledge
We thank the authors of [TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE) and [papagei-foundation-model](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) for their high-quality work.
