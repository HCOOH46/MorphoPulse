import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from baselines.common import default_artifact_dir, read_json
from baselines.factory import BASELINE_NAMES, BaselineGeneratorAdapter
from evaluation.metrics import calculate_fid, calculate_inception_score
from evaluation.rocket_functions import apply_kernels, generate_kernels
from evaluation.stat_metrics import (
    auto_correlation_difference,
    kurtosis_difference,
    marginal_distribution_difference,
    skewness_difference,
)
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
        self.use_custom_dataset = use_custom_dataset
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
        self._load_dataset(dataset_name, use_custom_dataset, config, test, target_length=input_length)
        self.ts_len = self.X_train.shape[-1]
        self.n_classes = len(np.unique(self.Y_train))
        print(f'n_classes: {self.n_classes}')

        self.generator_adapter = None
        self.stage2 = None
        self.maskgit = None
        self.stage1 = None
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
        z_test = remove_outliers(self.z_test)
        z_transform_pca = self.pca.fit_transform(z_test)
        self.xmin_pca, self.xmax_pca = np.min(z_transform_pca[:, 0]), np.max(z_transform_pca[:, 0])
        self.ymin_pca, self.ymax_pca = np.min(z_transform_pca[:, 1]), np.max(z_transform_pca[:, 1])

    def _load_feature_extractor(self, input_length: int, rocket_num_kernels: int):
        assert self.feature_extractor_type in ['papagei', 'supervised_fcn', 'rocket', 'PM'], 'unavailable feature extractor type.'
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

    def _load_dataset(self, dataset_name: str, use_custom_dataset: bool, config: dict, test: bool, target_length: int):
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset']) if not use_custom_dataset else (DatasetImporterCustomTest(dataset_name) if test else DatasetImporterCustom(dataset_name, **config['dataset']))
        self.X_train = dataset_importer.X_train
        self.X_test = dataset_importer.X_test
        self.Y_train = dataset_importer.Y_train
        self.Y_test = dataset_importer.Y_test
        self.mean = dataset_importer.mean
        self.std = dataset_importer.std
        if self.X_train.shape[-1] != target_length:
            print(f'Resampling evaluation dataset from length {self.X_train.shape[-1]} to {target_length}')
            self.X_train = self._resample_bcl(self.X_train, target_length)
            self.X_test = self._resample_bcl(self.X_test, target_length)

    @torch.no_grad()
    def sample(self, n_samples: int, kind: str, class_index: Union[int, None] = None, unscale: bool = False, batch_size=None):
        assert kind in ['unconditional', 'conditional']
        if self.generator_adapter is not None:
            if kind == 'unconditional':
                x_new = self.generator_adapter.sample_unconditional(n_samples, batch_size=batch_size)
            else:
                x_new = self.generator_adapter.sample_conditional(n_samples, class_index, batch_size=batch_size)
            x_new_l = None
            x_new_h = None
            X_new_R = x_new.clone()
        else:
            if kind == 'unconditional':
                x_new_l, x_new_h, x_new = unconditional_sample(self.maskgit, n_samples, self.device, batch_size=batch_size if batch_size is not None else self.batch_size)
            else:
                x_new_l, x_new_h, x_new = conditional_sample(self.maskgit, n_samples, self.device, class_index, batch_size=batch_size if batch_size is not None else self.batch_size)
            num_batches = x_new.shape[0] // self.batch_size + (1 if x_new.shape[0] % self.batch_size != 0 else 0)
            mapped = []
            for i in range(num_batches):
                s = slice(i * self.batch_size, (i + 1) * self.batch_size)
                mapped.append(self.neural_mapper(x_new[s].to(self.device)).cpu())
            X_new_R = torch.cat(mapped)

        if unscale:
            if x_new_l is not None:
                x_new_l = x_new_l * self.std + self.mean
                x_new_h = x_new_h * self.std + self.mean
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

    def compute_z_rec(self, kind: str):
        if not self.has_reconstruction_model:
            return self.compute_z(kind)
        X = self.X_train if kind == 'train' else self.X_test
        zs = []
        for i in range(0, X.shape[0], self.batch_size):
            x = torch.from_numpy(X[i:i + self.batch_size]).float().to(self.device)
            x_rec = self.stage1.forward(batch=(x, None), batch_idx=-1, return_x_rec=True).cpu().detach().numpy().astype(float)
            zs.append(self._extract_feature_representations(x_rec))
        return np.concatenate(zs, axis=0)

    @torch.no_grad()
    def compute_z_svq(self, kind: str):
        if not self.has_reconstruction_model or not hasattr(self.neural_mapper, 'tau'):
            raise RuntimeError('SVQ is only available for TimeVQVAE with a trained neural mapper.')
        X = self.X_train if kind == 'train' else self.X_test
        zs, xs_a = [], []
        for i in range(0, X.shape[0], self.batch_size):
            x = torch.from_numpy(X[i:i + self.batch_size]).float().to(self.device)
            tau = self.neural_mapper.tau.item()
            _, s_a_l = self.maskgit.encode_to_z_q(x, self.stage1.encoder_l, self.stage1.vq_model_l, svq_temp=tau)
            _, s_a_h = self.maskgit.encode_to_z_q(x, self.stage1.encoder_h, self.stage1.vq_model_h, svq_temp=tau)
            x_a = self.maskgit.decode_token_ind_to_timeseries(s_a_l, 'lf') + self.maskgit.decode_token_ind_to_timeseries(s_a_h, 'hf')
            x_a = x_a.cpu().numpy().astype(float)
            xs_a.append(x_a)
            zs.append(self._extract_feature_representations(x_a))
        return np.concatenate(zs, axis=0), np.concatenate(xs_a, axis=0)

    def compute_z(self, kind: str) -> np.ndarray:
        X = self.X_train if kind == 'train' else self.X_test
        zs = []
        for i in range(0, X.shape[0], self.batch_size):
            zs.append(self._extract_feature_representations(X[i:i + self.batch_size]))
        return np.concatenate(zs, axis=0)

    def compute_z_gen(self, X_gen: torch.Tensor) -> np.ndarray:
        z_gen = []
        for i in range(0, X_gen.shape[0], self.batch_size):
            z_gen.append(self._extract_feature_representations(X_gen[i:i + self.batch_size].numpy().astype(float)))
        return np.concatenate(z_gen, axis=0)

    def fid_score(self, z1: np.ndarray, z2: np.ndarray) -> int:
        return calculate_fid(remove_outliers(z1), remove_outliers(z2))

    def inception_score(self, X_gen: torch.Tensor):
        if self.fcn is None:
            raise RuntimeError('Inception score requires a classifier-like feature extractor.')
        p_yx_gen = []
        for i in range(0, self.X_test.shape[0], self.batch_size):
            s = slice(i, i + self.batch_size)
            p_yx_g = self.fcn(X_gen[s].float().to(self.device))
            p_yx_gen.append(torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy())
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)
        return calculate_inception_score(p_yx_gen)

    def stat_metrics(self, x_real: np.ndarray, x_gen: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            marginal_distribution_difference(x_real, x_gen),
            auto_correlation_difference(x_real, x_gen),
            skewness_difference(x_real, x_gen),
            kurtosis_difference(x_real, x_gen),
        )

    def log_visual_inspection(self, X1, X2, title: str, ylim: tuple = (-5, 5), n_plot_samples: int = 200, alpha: float = 0.1):
        _, c, _ = X1.shape
        sample_ind = np.random.randint(0, X1.shape[0], n_plot_samples)
        fig, axes = plt.subplots(2, c, figsize=(c * 4, 4))
        if c == 1:
            axes = axes[:, np.newaxis]
        plt.suptitle(title)
        for channel_idx in range(c):
            for i in sample_ind:
                axes[0, channel_idx].plot(X1[i, channel_idx, :], alpha=alpha, color='C0')
            axes[0, channel_idx].set_ylim(*ylim)
            axes[0, channel_idx].set_title(f'channel idx:{channel_idx}')
            sample_ind = np.random.randint(0, X2.shape[0], n_plot_samples)
            for i in sample_ind:
                axes[1, channel_idx].plot(X2[i, channel_idx, :], alpha=alpha, color='C0')
            axes[1, channel_idx].set_ylim(*ylim)
            if channel_idx == 0:
                axes[0, channel_idx].set_ylabel('X_test')
                axes[1, channel_idx].set_ylabel('X_gen')
        plt.tight_layout()
        wandb.log({f'visual comp ({title})': wandb.Image(plt)})
        plt.close()

    def log_pca(self, Zs: List[np.ndarray], labels: List[str], n_plot_samples: int = 1000):
        plt.figure(figsize=(4, 4))
        for Z, label in zip(Zs, labels):
            ind = np.random.choice(range(Z.shape[0]), size=n_plot_samples, replace=True)
            Z_embed = self.pca.transform(Z[ind])
            plt.scatter(Z_embed[:, 0], Z_embed[:, 1], alpha=0.1, label=label)
            xpad = (self.xmax_pca - self.xmin_pca) * 0.1
            ypad = (self.ymax_pca - self.ymin_pca) * 0.1
            plt.xlim(self.xmin_pca - xpad, self.xmax_pca + xpad)
            plt.ylim(self.ymin_pca - ypad, self.ymax_pca + ypad)
        plt.tight_layout()
        wandb.log({f"PCA on Z ({'-'.join(labels)})": wandb.Image(plt)})
        plt.close()

    def log_tsne(self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray):
        X_gen = F.interpolate(X_gen, size=self.X_test.shape[-1], mode='linear', align_corners=True).cpu().numpy()
        sample_ind_test = np.random.randint(0, self.X_test.shape[0], n_plot_samples)
        sample_ind_gen = np.random.randint(0, X_gen.shape[0], n_plot_samples)
        X = np.concatenate((self.X_test.squeeze()[sample_ind_test], X_gen.squeeze()[sample_ind_gen]), axis=0).squeeze()
        labels = np.array(['C0'] * len(sample_ind_test) + ['C1'] * len(sample_ind_gen))
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X)
        plt.figure(figsize=(4, 4))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, alpha=0.1)
        plt.tight_layout()
        wandb.log({'TNSE-data_space': wandb.Image(plt)})
        plt.close()
        Z = np.concatenate((z_test[sample_ind_test], z_gen[sample_ind_gen]), axis=0).squeeze()
        Z_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(Z)
        plt.figure(figsize=(4, 4))
        plt.scatter(Z_embedded[:, 0], Z_embedded[:, 1], c=labels, alpha=0.1)
        plt.tight_layout()
        wandb.log({'TSNE-latent_space': wandb.Image(plt)})
        plt.close()
