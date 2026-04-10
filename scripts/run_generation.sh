#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PROJECT_ROOT
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/artifacts/checkpoints}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/outputs}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/foundation:$PROJECT_ROOT/src/generation:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/config_gen.yaml}"
DATASET_NAME="${DATASET_NAME:-Mesa_ppg_svri}"
GPU_DEVICE_IND="${GPU_DEVICE_IND:-0}"
N_SAMPLES="${N_SAMPLES:-5000000}"

cd "$PROJECT_ROOT/src/generation"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python stage1.py --dataset_names "$DATASET_NAME" --gpu_device_ind "$GPU_DEVICE_IND" --use_custom_dataset True --config "$CONFIG_PATH"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python stage2.py --dataset_names "$DATASET_NAME" --gpu_device_ind "$GPU_DEVICE_IND" --use_custom_dataset True --config "$CONFIG_PATH"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python evaluate.py --dataset_names "$DATASET_NAME" --gpu_device_ind "$GPU_DEVICE_IND" --use_custom_dataset True --config "$CONFIG_PATH"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python evaluate_v.py --dataset_names "$DATASET_NAME" --gpu_device_ind "$GPU_DEVICE_IND" --use_custom_dataset True --config "$CONFIG_PATH" --feature_extractor_type papagei
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python sample.py --dataset_names "$DATASET_NAME" --gpu_device_ind "$GPU_DEVICE_IND" --use_custom_dataset True --config "$CONFIG_PATH" --n_samples "$N_SAMPLES"

