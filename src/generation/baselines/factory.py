from typing import Optional

import numpy as np
import torch

from .common import choose_samples, default_artifact_dir, load_sample_bank


BASELINE_REGISTRY = {
    'mctimegan': 'MC-TimeGAN (official)',
    'diffusion_ts': 'Diffusion-TS (official)',
    'timedp': 'TimeDP (official)',
}
BASELINE_NAMES = tuple(BASELINE_REGISTRY.keys())


class BaselineGeneratorAdapter:
    def __init__(
        self,
        generator_name: str,
        input_length: int,
        n_classes: int,
        config: dict,
        device,
        ckpt_path: Optional[str] = None,
        sampling_steps: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        del config, sampling_steps
        self.generator_name = generator_name.lower()
        self.device = torch.device(device)
        if self.generator_name not in BASELINE_NAMES:
            raise ValueError(f'Unknown official baseline: {self.generator_name}')
        if ckpt_path is None:
            if dataset_name is None:
                raise ValueError('dataset_name is required when generator_ckpt is omitted.')
            ckpt_path = default_artifact_dir(dataset_name, self.generator_name)
        self.sample_bank = load_sample_bank(ckpt_path, expected_n_classes=n_classes)
        bank_input_length = int(self.sample_bank['metadata']['input_length'])
        if bank_input_length != int(input_length):
            raise ValueError(f'Sample bank length {bank_input_length} does not match expected {input_length}.')
        self.rng = np.random.default_rng(0)

    @torch.no_grad()
    def sample_unconditional(self, n_samples: int, batch_size: Optional[int] = None) -> torch.Tensor:
        del batch_size
        bank = self.sample_bank['unconditional']
        if bank is None:
            raise RuntimeError('Unconditional sample bank is missing.')
        return torch.from_numpy(choose_samples(bank, n_samples, self.rng)).float()

    @torch.no_grad()
    def sample_conditional(self, n_samples: int, class_index: int, batch_size: Optional[int] = None) -> torch.Tensor:
        del batch_size
        bank = self.sample_bank['conditional'][int(class_index)]
        return torch.from_numpy(choose_samples(bank, n_samples, self.rng)).float()
