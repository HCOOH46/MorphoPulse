# MorphoPulse
Open-source repository of MorphoPulse, a morphology‑controlled pre‑training framework that densifies sparse regions of the PPG physiological manifold through clinically guided synthetic data generation

## Released Assets

- Final checkpoint: `artifacts/checkpoints/foundation/MorphoPulse/MorphoPulse.pt`
- Final checkpoint size: `47,583,353` bytes
- Final checkpoint checksum: `a904444a30bddc54c3dd41be5ac8a43fadc355bb8fa8aab8ae1cfeebd2041585`
- Matching synthetic pretraining split: `g`
- Flattened synthetic split stats:
  - `VitalDB_g`: `1,943,079` segments
  - `MESA_g`: `4,514,963` segments
  - `MIMIC_g`: `1,964,013` segments
  - Total: `8,422,055` segments
  - Per-segment shape: `float32[2560]`
  - Flattened `.npy` size: about `80.3 GiB`

The released `g` synthetic set is the first-generation synthetic data only. Later `g2` and `g2_1` resampled variants are intentionally excluded from this release.

## Repository layout

- `src/foundation/`: foundation model training and evaluation code
- `src/generation/`: synthetic generation code
- `configs/`: released configs
- `docs/`: release notes and publishing plan
- `scripts/`: runnable entry scripts
- `datasets/`: placeholder directory structure only

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

The public synthetic release should be published as one flattened `MorphoPulse_g_synthetic.npy` file instead of virtual-patient folders. GitHub is suitable for the code and checkpoint, but not for the flattened synthetic array. A public dataset host that accepts very large artifacts, such as a Hugging Face dataset repository, is the right target for that file.

Recommended public targets:

- Code and checkpoint: `https://github.com/HCOOH46/MorphoPulse`
- Synthetic dataset: `https://huggingface.co/datasets/HCOOH46/MorphoPulse-g-synthetic`

## Checkpoints

The released checkpoint is stored in-repo at `artifacts/checkpoints/foundation/MorphoPulse/MorphoPulse.pt`.

Example evaluation command:

```bash
MODEL=MorphoPulse.pt TIME=MorphoPulse bash scripts/eval_foundation.sh
```

## Synthetic Export

To flatten the released `g` synthetic split into a single `.npy` file, run:

```bash
python scripts/export_morphopulse_g_dataset.py \
  --data-root /path/to/datasets \
  --output-npy /path/to/MorphoPulse_g_synthetic.npy \
  --output-meta /path/to/MorphoPulse_g_synthetic_meta.csv
```

The exporter preserves the exact filtering used by `harmonize_datasets(dataset_name="g")` and removes the virtual-patient directory structure in the final array artifact.

Detailed release notes are in `docs/OPEN_SOURCE_RELEASE.md`.

# Acknowledge
We thank the authors of [TimeVQVAE](https://github.com/ML4ITS/TimeVQVAE) and [papagei-foundation-model](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) for their high-quality work.
