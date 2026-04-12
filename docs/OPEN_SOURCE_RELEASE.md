# MorphoPulse Open-Source Release Plan

## Decision

Use two public endpoints:

- Code and final checkpoint: `https://github.com/HCOOH46/MorphoPulse`
- Flattened synthetic `g` dataset: `https://huggingface.co/datasets/HCOOH46/MorphoPulse-g-synthetic`

This split matches the asset sizes and the hosting limits of the two platforms.

## Why This Split

- GitHub blocks files larger than 100 MiB in a repository.
- GitHub release assets must each stay under 2 GiB.
- `MorphoPulse.pt` is only `47,583,353` bytes, so it fits directly in the GitHub repository.
- The flattened `g` synthetic dataset is about `80.3 GiB` as a single `float32` NumPy array, so it does not fit GitHub repository storage or GitHub release assets.
- Hugging Face Hub allows very large dataset artifacts. Its storage guidance says a single file can be as large as 500 GB, with smaller files preferred operationally. Your requirement is one large `.npy`, and this release fits within that hard limit.

Reference docs:

- GitHub large file limits: `https://docs.github.com/repositories/working-with-files/managing-large-files/about-large-files-on-github`
- GitHub release asset limits: `https://docs.github.com/articles/getting-the-download-count-for-your-releases`
- Hugging Face Hub storage limits: `https://huggingface.co/docs/hub/en/storage-limits`

## Released Model

- Public filename: `MorphoPulse.pt`
- Repository path: `artifacts/checkpoints/foundation/MorphoPulse/MorphoPulse.pt`
- SHA256: `a904444a30bddc54c3dd41be5ac8a43fadc355bb8fa8aab8ae1cfeebd2041585`
- Source checkpoint:
  `/data0/zjh/models/2025_11_29_17_21_05/resnet_mt_moe_18_g__kwdjiu_2025_11_29_17_21_05_step4554_loss1.5228.pt`

## Released Synthetic Dataset

Only the first-batch synthetic `g` split should be public.

- Include:
  - `VitalDB_g`
  - `MESA_g`
  - `MIMIC_g`
- Exclude:
  - `g2`
  - `g2_1`
  - any later resampled synthetic variants

After applying the same filtering used in `harmonize_datasets(dataset_name="g")`, the public synthetic release contains:

- `VitalDB_g`: `1,943,079` segments
- `MESA_g`: `4,514,963` segments
- `MIMIC_g`: `1,964,013` segments
- Total: `8,422,055` segments

Each segment is:

- dtype: `float32`
- shape: `(2560,)`

The flattened public artifact should be:

- `MorphoPulse_g_synthetic.npy`
- optional sidecar metadata: `MorphoPulse_g_synthetic_meta.csv`

## Export Command

Run from this repository after placing the original metadata CSVs and generated segment folders under `datasets/`:

```bash
python scripts/export_morphopulse_g_dataset.py \
  --data-root /path/to/morphopulse-open/datasets \
  --output-npy /path/to/MorphoPulse_g_synthetic.npy \
  --output-meta /path/to/MorphoPulse_g_synthetic_meta.csv
```

The exporter:

- preserves the exact `g` filtering logic
- removes the virtual-patient directory structure
- emits one large `.npy`
- optionally emits flattened metadata for traceability

## Suggested Hugging Face Dataset Layout

Recommended repository: `HCOOH46/MorphoPulse-g-synthetic`

Suggested top-level files:

- `MorphoPulse_g_synthetic.npy`
- `MorphoPulse_g_synthetic_meta.csv`
- `README.md`
- `SHA256SUMS`

Suggested dataset card points:

- this is synthetic PPG pretraining data only
- it corresponds to the released `MorphoPulse.pt` checkpoint
- it uses the first-batch `g` synthetic data only
- no real patient waveforms are redistributed in this artifact

## Local Status

This repository has already been prepared locally with:

- the final checkpoint copied into `artifacts/checkpoints/foundation/MorphoPulse/`
- the synthetic exporter script in `scripts/export_morphopulse_g_dataset.py`
- README updates describing the release split

No remote commit or push is required from the agent side.
