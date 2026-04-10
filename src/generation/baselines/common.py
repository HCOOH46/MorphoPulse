import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from project_paths import GENERATION_CHECKPOINT_ROOT


def default_baseline_configs() -> Dict[str, dict]:
    return {
        'mctimegan': {
            'epochs': 500,
            'batch_size': 128,
            'sample_batch_size': 512,
            'hidden_dim': 24,
            'num_layers': 3,
            'learning_rate': 1e-4,
            'reproducibility': True,
            'max_train_samples': 20000,
            'grad_clip': 1.0,
            'sqrt_eps': 1e-8,
        },
        'diffusion_ts': {
            'max_epochs': 4000,
            'save_cycle': 400,
            'batch_size': 2,
            'sample_batch_size': 8,
            'base_lr': 1e-4,
            'gradient_accumulate_every': 8,
            'log_every_steps': 100,
            'timesteps': 500,
            'sampling_timesteps': 100,
            'd_model': 64,
            'n_layer_enc': 3,
            'n_layer_dec': 2,
            'n_heads': 4,
            'mlp_hidden_times': 4,
            'size_every': 1001,
            'max_train_samples': 20000,
        },
        'timedp': {
            'max_steps': 20000,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'seed': 23,
            'num_latents': 16,
            'logdir': 'logs_official_baselines',
            'max_train_samples': 20000,
        },
    }


def merge_baseline_config(config: dict, generator_name: str) -> dict:
    merged = default_baseline_configs()[generator_name].copy()
    merged.update(config.get('baselines', {}).get(generator_name, {}))
    return merged


def default_artifact_dir(dataset_name: str, generator_name: str) -> str:
    return os.path.join(GENERATION_CHECKPOINT_ROOT, f'baseline-{generator_name}-{dataset_name}')


def ensure_bcl(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    elif arr.ndim != 3:
        raise ValueError(f'Expected a 2D/3D array, got shape={arr.shape}.')

    if arr.shape[1] == 1:
        return arr
    if arr.shape[2] == 1:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f'Unable to coerce array with shape={arr.shape} to (B, 1, L).')


def choose_samples(array: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if len(array) == 0:
        raise ValueError('No samples are available in the requested sample bank.')
    replace = len(array) < n_samples
    indices = rng.choice(len(array), size=n_samples, replace=replace)
    return np.asarray(array[indices], dtype=np.float32)


def write_json(path: str, data: Mapping):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_sample_bank(path: str, expected_n_classes: Optional[int] = None) -> dict:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    metadata_path = os.path.join(path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'Expected metadata at {metadata_path}.')
    metadata = read_json(metadata_path)
    n_classes = int(metadata['n_classes'])
    if expected_n_classes is not None and n_classes != expected_n_classes:
        raise ValueError(f'Sample bank has {n_classes} classes, expected {expected_n_classes}.')
    conditional = {}
    for cls_idx in range(n_classes):
        cls_path = os.path.join(path, f'conditional_cls{cls_idx}.npy')
        if not os.path.exists(cls_path):
            raise FileNotFoundError(f'Missing conditional sample bank: {cls_path}')
        conditional[cls_idx] = ensure_bcl(np.load(cls_path))
    unconditional_path = os.path.join(path, 'unconditional.npy')
    unconditional = ensure_bcl(np.load(unconditional_path)) if os.path.exists(unconditional_path) else None
    return {
        'artifact_dir': path,
        'metadata': metadata,
        'conditional': conditional,
        'unconditional': unconditional,
    }


def save_sample_bank(artifact_dir: str, conditional: Mapping[int, np.ndarray], unconditional: np.ndarray, metadata: Mapping):
    os.makedirs(artifact_dir, exist_ok=True)
    for cls_idx, samples in conditional.items():
        np.save(os.path.join(artifact_dir, f'conditional_cls{int(cls_idx)}.npy'), ensure_bcl(samples))
    np.save(os.path.join(artifact_dir, 'unconditional.npy'), ensure_bcl(unconditional))
    write_json(os.path.join(artifact_dir, 'metadata.json'), dict(metadata))


def mix_unconditional_from_conditional(
    conditional: Mapping[int, np.ndarray],
    class_probs: np.ndarray,
    total_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_ids = np.arange(len(class_probs))
    draws = rng.choice(class_ids, size=total_samples, p=class_probs)
    buffers = []
    for cls_idx in class_ids:
        count = int(np.sum(draws == cls_idx))
        if count == 0:
            continue
        buffers.append(choose_samples(conditional[int(cls_idx)], count, rng))
    if not buffers:
        raise ValueError('Unable to create unconditional samples from empty class buffers.')
    unconditional = np.concatenate(buffers, axis=0)
    rng.shuffle(unconditional)
    return unconditional.astype(np.float32)


def to_torch_bcl(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ensure_bcl(array)).float()
