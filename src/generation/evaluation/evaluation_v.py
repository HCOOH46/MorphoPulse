import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from baselines.common import default_artifact_dir, read_json
from baselines.factory import BASELINE_NAMES, BaselineGeneratorAdapter
from evaluation.metrics import calculate_fid
from evaluation.rocket_functions import apply_kernels, generate_kernels
from evaluation.stat_metrics import auto_correlation_difference, kurtosis_difference, marginal_distribution_difference, skewness_difference
from experiments.exp_stage2 import ExpStage2
from generators.neural_mapper import NeuralMapper
from generators.sample import conditional_sample, unconditional_sample
from preprocessing.preprocess_ucr import DatasetImporterCustom, DatasetImporterCustomTest, DatasetImporterUCR
from utils import remove_outliers
from project_paths import GENERATION_CHECKPOINT_ROOT


class Evaluation(nn.Module):
    def __init__(
        self,
        dataset_name: str,
        in_channels: int,
        input_length: int,
        n_classes: int,
        device,
        config: dict,
        use_neural_mapper: bool = False,
        feature_extractor_type: str = 'rocket',
        rocket_num_kernels: int = 1000,
        use_custom_dataset: bool = False,
        test: bool = False,
        DDiT: bool = False,
        generator: str = 'timevqvae',
        generator_ckpt: Union[str, None] = None,
        sampling_steps: Union[int, None] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        if isinstance(device, int):
            device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.config = config
        self.batch_size = self.config['evaluation']['batch_size']
        self.feature_extractor_type = feature_extractor_type
        self.generator_name = generator.lower()
        self.save_path = Path(f'./visual/{dataset_name}')
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.has_reconstruction_model = self.generator_name == 'timevqvae'
        self.supports_neural_mapper = self.has_reconstruction_model
        if self.generator_name not in ('timevqvae',) + BASELINE_NAMES:
            raise ValueError(f'Unsupported generator: {self.generator_name}')
        if use_neural_mapper and not self.supports_neural_mapper:
            print(f'Neural mapper is unavailable for generator={self.generator_name}. Falling back to identity mapping.')
            use_neural_mapper = False

        if self.generator_name != 'timevqvae':
            if generator_ckpt is None:
                default_ckpt = default_artifact_dir(dataset_name, self.generator_name)
                generator_ckpt = default_ckpt if os.path.exists(default_ckpt) else None
            if generator_ckpt is not None:
                metadata_path = os.path.join(generator_ckpt, 'metadata.json') if os.path.isdir(generator_ckpt) else os.path.join(os.path.dirname(generator_ckpt), 'metadata.json')
                if os.path.exists(metadata_path):
                    input_length = int(read_json(metadata_path)['input_length'])
                    print(f'Using baseline artifact input_length={input_length} from {metadata_path}')

        self._load_feature_extractor(input_length, rocket_num_kernels)
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset']) if not use_custom_dataset else (DatasetImporterCustomTest(dataset_name) if test else DatasetImporterCustom(dataset_name, **config['dataset']))
        self.X_train = dataset_importer.X_train
        self.X_test = dataset_importer.X_test
        self.Y_train = dataset_importer.Y_train
        self.Y_test = dataset_importer.Y_test
        self.mean = dataset_importer.mean
        self.std = dataset_importer.std
        if self.X_train.shape[-1] != input_length:
            print(f'Resampling evaluation dataset from length {self.X_train.shape[-1]} to {input_length}')
            self.X_train = self._resample_bcl(self.X_train, input_length)
            self.X_test = self._resample_bcl(self.X_test, input_length)
        self.ts_len = self.X_train.shape[-1]
        self.n_classes = len(np.unique(dataset_importer.Y_train))

        self.generator_adapter = None
        if self.generator_name == 'timevqvae':
            save_file = f'stage2-{dataset_name}-DDiT.ckpt' if DDiT else f'stage2-{dataset_name}.ckpt'
            self.stage2 = ExpStage2.load_from_checkpoint(
                os.path.join(GENERATION_CHECKPOINT_ROOT, save_file),
                dataset_name=dataset_name,
                in_channels=in_channels,
                input_length=input_length,
                config=config,
                n_classes=n_classes,
                feature_extractor_type=feature_extractor_type,
                test=test,
                diffusion=DDiT,
                use_custom_dataset=use_custom_dataset,
                map_location='cpu',
                strict=False,
            )
            self.stage2.eval()
            self.maskgit = self.stage2.stage2
            self.stage1 = self.maskgit.stage1
        else:
            if generator_ckpt is None:
                default_ckpt = default_artifact_dir(dataset_name, self.generator_name)
                generator_ckpt = default_ckpt if os.path.exists(default_ckpt) else None
            self.stage2 = None
            self.maskgit = None
            self.stage1 = None
            self.generator_adapter = BaselineGeneratorAdapter(
                self.generator_name,
                input_length=input_length,
                n_classes=n_classes,
                config=config,
                device=self.device,
                ckpt_path=generator_ckpt,
                sampling_steps=sampling_steps,
                dataset_name=dataset_name,
            )

        if use_neural_mapper:
            self.neural_mapper = NeuralMapper(self.ts_len, 1, config)
            fname = f'neural_mapper-{dataset_name}.ckpt'
            self.neural_mapper.load_state_dict(torch.load(os.path.join(GENERATION_CHECKPOINT_ROOT, fname)))
        else:
            self.neural_mapper = nn.Identity()

        self.pca = PCA(n_components=2, random_state=0)
        self.z_train = self.compute_z('train')
        self.z_test = self.compute_z('test')
        z_test_clean = remove_outliers(self.z_test)
        self.pca.fit(z_test_clean)
        z_transform_pca = self.pca.transform(z_test_clean)
        self.xmin_pca, self.xmax_pca = np.min(z_transform_pca[:, 0]), np.max(z_transform_pca[:, 0])
        self.ymin_pca, self.ymax_pca = np.min(z_transform_pca[:, 1]), np.max(z_transform_pca[:, 1])

    def _load_feature_extractor(self, input_length: int, rocket_num_kernels: int):
        self.fcn = None
        if self.feature_extractor_type == 'rocket':
            self.rocket_kernels = generate_kernels(input_length, num_kernels=rocket_num_kernels)
            return
        if self.feature_extractor_type in ['papagei', 'PM']:
            from outside_call import get_papagei, get_param
            if self.feature_extractor_type == 'papagei':
                fcn0 = get_papagei()
                self.fcn = get_param(fcn0)
            else:
                fcn0 = get_papagei('Mamba1')
                self.fcn = get_param(fcn0, 'Mamba1')
            self.fcn.to(self.device)
            self.fcn.eval()
            return
        raise NotImplementedError('supervised_fcn feature extractor is not configured in this environment.')

    @staticmethod
    def _resample_bcl(array: np.ndarray, target_length: int) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        if arr.shape[-1] == target_length:
            return arr
        old_length = arr.shape[-1]
        old_idx = np.linspace(0, old_length - 1, old_length, dtype=np.float32)
        new_idx = np.linspace(0, old_length - 1, target_length, dtype=np.float32)
        out = np.empty((arr.shape[0], arr.shape[1], target_length), dtype=np.float32)
        for i in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                out[i, c] = np.interp(new_idx, old_idx, arr[i, c]).astype(np.float32)
        return out

    def plot_tsne_styled(self, z_real: np.ndarray, z_gen: np.ndarray, filename: str, n_plot_samples: int = 1000):
        n_real = min(len(z_real), n_plot_samples)
        n_gen = min(len(z_gen), n_plot_samples)
        idx_real = np.random.choice(len(z_real), n_real, replace=False)
        idx_gen = np.random.choice(len(z_gen), n_gen, replace=False)
        z_combined = np.concatenate([z_real[idx_real], z_gen[idx_gen]], axis=0)
        z_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z_combined)
        plt.figure(figsize=(6, 5))
        plt.scatter(z_2d[:n_real, 0], z_2d[:n_real, 1], c='#4682B4', alpha=0.5, label='Real', s=15, edgecolors='none')
        plt.scatter(z_2d[n_real:, 0], z_2d[n_real:, 1], c='#FA8072', alpha=0.5, label='Generated', s=15, edgecolors='none')
        plt.title(f't-SNE Space Visualization ({self.dataset_name})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path / f'{filename}.png', dpi=300)
        plt.close()

    def plot_hist_cdf(self, z_real: np.ndarray, z_gen: np.ndarray, filename: str):
        real_pc1 = self.pca.transform(z_real)[:, 0]
        gen_pc1 = self.pca.transform(z_gen)[:, 0]
        combined = np.concatenate([real_pc1, gen_pc1])
        vmin, vmax = np.percentile(combined, [10, 90])
        margin = (vmax - vmin) * 0.05
        plot_range = (vmin - margin, vmax + margin)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.linewidth'] = 0.8
        color_real, color_gen = '#8ab5d3', '#f4a492'
        line_real, line_gen = '#4682b4', '#e36e57'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        num_bins = 35
        ax1.hist(real_pc1, bins=num_bins, range=plot_range, alpha=0.6, label='Real', density=True, color=color_real, ec='white', lw=0.5)
        ax1.hist(gen_pc1, bins=num_bins, range=plot_range, alpha=0.5, label='Generated', density=True, color=color_gen, ec='white', lw=0.5)
        ax1.set_title('Feature Distribution (PC1)', fontsize=12, fontweight='bold', pad=15)
        ax1.set_xlabel('Feature value range', fontsize=10)
        ax1.set_ylabel('Density', fontsize=10)
        ax1.legend(frameon=False, loc='upper right', fontsize=9)
        counts_real, bin_edges = np.histogram(real_pc1, bins=num_bins, range=plot_range)
        counts_gen, _ = np.histogram(gen_pc1, bins=num_bins, range=plot_range)
        cdf_real = np.cumsum(counts_real) / max(counts_real.sum(), 1)
        cdf_gen = np.cumsum(counts_gen) / max(counts_gen.sum(), 1)
        x_steps = bin_edges
        y_real = np.insert(cdf_real, 0, 0)
        y_gen = np.insert(cdf_gen, 0, 0)
        ax2.step(x_steps, y_real, label='Real', color=line_real, where='post', lw=1.8, alpha=0.9)
        ax2.step(x_steps, y_gen, label='Generated', color=line_gen, where='post', lw=1.8, alpha=0.9)
        ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold', pad=15)
        ax2.set_xlabel('Feature value range', fontsize=10)
        ax2.set_ylabel('Cumulative probability', fontsize=10)
        ax2.legend(frameon=False, loc='lower right', fontsize=9)
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(plot_range)
            ax.tick_params(direction='out', length=4, width=0.8)
        plt.tight_layout()
        plt.savefig(self.save_path / f'{filename}.png', bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def sample(self, n_samples: int, kind: str, class_index: Union[int, None] = None, unscale: bool = False, batch_size=None):
        assert kind in ['unconditional', 'conditional']
        if self.generator_adapter is not None:
            x_new = self.generator_adapter.sample_unconditional(n_samples, batch_size=batch_size) if kind == 'unconditional' else self.generator_adapter.sample_conditional(n_samples, class_index, batch_size=batch_size)
            x_new_l, x_new_h = None, None
            X_new_R = x_new.clone()
        else:
            if kind == 'unconditional':
                x_new_l, x_new_h, x_new = unconditional_sample(self.maskgit, n_samples, self.device, batch_size=batch_size if batch_size else self.batch_size)
            else:
                x_new_l, x_new_h, x_new = conditional_sample(self.maskgit, n_samples, self.device, class_index, batch_size=batch_size if batch_size else self.batch_size)
            num_batches = (x_new.shape[0] + self.batch_size - 1) // self.batch_size
            X_new_R = []
            for i in range(num_batches):
                mini_batch = x_new[i * self.batch_size:(i + 1) * self.batch_size]
                X_new_R.append(self.neural_mapper(mini_batch.to(self.device)).cpu())
            X_new_R = torch.cat(X_new_R)
        if unscale:
            x_new = x_new * self.std + self.mean
            X_new_R = X_new_R * self.std + self.mean
        return (x_new_l, x_new_h, x_new), X_new_R

    def _extract_feature_representations(self, x: np.ndarray):
        if self.feature_extractor_type in ['papagei', 'PM']:
            z = self.fcn(torch.from_numpy(x).float().to(self.device))[0].cpu().detach().numpy()
        elif self.feature_extractor_type == 'supervised_fcn':
            z = self.fcn(torch.from_numpy(x).float().to(self.device), return_feature_vector=True).cpu().detach().numpy()
        elif self.feature_extractor_type == 'rocket':
            z = apply_kernels(np.asarray(x[:, 0, :], dtype=np.float64), self.rocket_kernels)
            z = F.normalize(torch.from_numpy(z), p=2, dim=1).numpy()
        else:
            raise ValueError
        return z

    def compute_z(self, kind: str) -> np.ndarray:
        X = self.X_train if kind == 'train' else self.X_test
        zs = []
        for i in range(0, X.shape[0], self.batch_size):
            zs.append(self._extract_feature_representations(X[i:i + self.batch_size]))
        return np.concatenate(zs, axis=0)

    def compute_z_gen(self, X_gen: torch.Tensor) -> np.ndarray:
        zs = []
        for i in range(0, X_gen.shape[0], self.batch_size):
            zs.append(self._extract_feature_representations(X_gen[i:i + self.batch_size].numpy().astype(float)))
        return np.concatenate(zs, axis=0)

    def fid_score(self, z1: np.ndarray, z2: np.ndarray) -> int:
        return calculate_fid(remove_outliers(z1), remove_outliers(z2))

    def stat_metrics(self, x_real, x_gen):
        return marginal_distribution_difference(x_real, x_gen), auto_correlation_difference(x_real, x_gen), skewness_difference(x_real, x_gen), kurtosis_difference(x_real, x_gen)

    def log_visual_inspection(self, X1, X2, title, ylim=(-5, 5), n_plot_samples=200, alpha=0.1):
        pass
