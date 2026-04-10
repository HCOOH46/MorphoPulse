# MorphoPulse
Open-source repository of MorphoPulse.

## Abstract
Foundation models for photoplethysmography (PPG)—a key signal for wearable and bedside monitoring—often fail to generalize to underrepresented waveform patterns and rare vascular phenotypes due to narrow training cohorts. To address this translational gap, we introduce MorphoPulse, a morphology‑controlled pre‑training framework that densifies sparse regions of the PPG physiological manifold through clinically guided synthetic data generation. MorphoPulse uses a vector‑quantized variational autoencoder conditioned on the stress‑induced vascular response index (sVRI), a validated descriptor of vascular dynamics, to synthesize high‑fidelity PPG signals that fill morphological gaps in real training data. An efficient Mamba‑based state‑space model (11M parameters) is pre‑trained on a 1:1 hybrid corpus of real and synthetic signals, with auxiliary geometric objectives that align representations with interpretable physiological dimensions. Across nine downstream tasks—including clinical diagnostics (atrial fibrillation, hypertension) and wearable wellness monitoring (mood recognition, activity classification)—MorphoPulse improves robustness to sensor hardware and acquisition shifts, achieving a 6.8% average AUROC gain over prior PPG‑specific foundation models and a 2.1% gain over general time‑series baselines. Ablation studies show that morphology‑guided synthesis and the Mamba backbone synergistically enhance out‑of‑distribution generalization, particularly for rare PPG morphologies where existing models fail. MorphoPulse demonstrates that embedding domain‑specific structural priors into hybrid pre‑training yields robust physiological foundation models suitable for edge deployment. All code, pre‑trained weights, and synthetic datasets are released as open‑source resources to accelerate translational PPG research.

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
