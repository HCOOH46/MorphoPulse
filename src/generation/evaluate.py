"""
Evaluate waveform generators on unconditional and label-conditional generation.
"""
import random
from argparse import ArgumentParser
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

from baselines.factory import BASELINE_NAMES
from evaluation.evaluation import Evaluation
from evcm import evaluate_conditional_generation_consistency
from ridge import evaluate_conditional_generation_consistency as evaluate_conditional_generation_consistency_ridge
from preprocessing.data_pipeline import build_custom_data_pipeline, build_data_pipeline
from preprocessing.preprocess_ucr import DatasetImporterCustomTest, DatasetImporterUCR
from utils import get_root_dir, load_yaml_param_settings, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', default=[])
    parser.add_argument('--gpu_device_ind', default=0, type=int)
    parser.add_argument('--use_neural_mapper', type=str2bool, default=False)
    parser.add_argument('--feature_extractor_type', type=str, default='papagei')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False)
    parser.add_argument('--sampling_batch_size', type=int, default=None)
    parser.add_argument('--generator', type=str, default='timevqvae', choices=['timevqvae', *BASELINE_NAMES])
    parser.add_argument('--generator_ckpt', type=str, default=None)
    parser.add_argument('--sampling_steps', type=int, default=None)
    parser.add_argument('--use_ridge_plots', type=str2bool, default=False)
    return parser.parse_args()


def compute_identity_baseline(evaluation, n_classes):
    print('\n--- Computing Identity Baseline (Real_A vs Real_B) ---')
    X_test = evaluation.X_test
    Y_test = evaluation.Y_test
    n_samples = X_test.shape[0]
    indices = np.random.permutation(n_samples)
    mid = n_samples // 2
    indices_A = indices[:mid]
    indices_B = indices[mid:2 * mid]
    X_A, X_B = X_test[indices_A], X_test[indices_B]
    Y_A, Y_B = Y_test[indices_A], Y_test[indices_B]
    z_A = evaluation.compute_z_gen(torch.from_numpy(X_A))
    z_B = evaluation.compute_z_gen(torch.from_numpy(X_B))
    baseline_fid = evaluation.fid_score(z_A, z_B)
    baseline_mdd, baseline_acd, baseline_sd, baseline_kd = evaluation.stat_metrics(torch.from_numpy(X_A), torch.from_numpy(X_B))
    baseline_cfids = []
    for cls_idx in range(n_classes):
        cls_ind_A = (Y_A == cls_idx)
        cls_ind_B = (Y_B == cls_idx)
        if cls_ind_A.sum() < 2 or cls_ind_B.sum() < 2:
            continue
        z_A_c = evaluation.compute_z_gen(torch.from_numpy(X_A[cls_ind_A]))
        z_B_c = evaluation.compute_z_gen(torch.from_numpy(X_B[cls_ind_B]))
        baseline_cfids.append(evaluation.fid_score(z_A_c, z_B_c))
    baseline_cfid_mean = np.mean(baseline_cfids) if baseline_cfids else float('nan')
    print(f'Baseline FID={baseline_fid:.4f} cFID={baseline_cfid_mean:.4f}')
    return {
        'FID': baseline_fid,
        'MDD': baseline_mdd,
        'ACD': baseline_acd,
        'SD': baseline_sd,
        'KD': baseline_kd,
        'cFID': baseline_cfid_mean,
        'cFIDs': baseline_cfids,
    }


def evaluate(
    config: dict,
    dataset_name: str,
    train_data_loader: DataLoader,
    gpu_device_idx,
    use_neural_mapper: bool,
    feature_extractor_type: str,
    use_custom_dataset: bool = False,
    sampling_batch_size=None,
    generator: str = 'timevqvae',
    generator_ckpt: Union[str, None] = None,
    sampling_steps: Union[int, None] = None,
    rand_seed: Union[int, None] = None,
    use_ridge_plots: bool = False,
):
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
    if generator != 'timevqvae':
        use_neural_mapper = False

    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    wandb.init(project='TimeVQVAE-evaluation', config={**config, 'dataset_name': dataset_name, 'generator': generator})
    evaluation = Evaluation(
        dataset_name,
        in_channels,
        input_length,
        n_classes,
        gpu_device_idx,
        config,
        use_neural_mapper=use_neural_mapper,
        feature_extractor_type=feature_extractor_type,
        use_custom_dataset=use_custom_dataset,
        test=True,
        generator=generator,
        generator_ckpt=generator_ckpt,
        sampling_steps=sampling_steps,
    ).to(gpu_device_idx)

    baseline_metrics = compute_identity_baseline(evaluation, n_classes)
    wandb.log({f'Baseline_{k}': v for k, v in baseline_metrics.items() if k != 'cFIDs'})

    min_num_gen_samples = config['evaluation']['min_num_gen_samples']
    (_, _, xhat), xhat_R = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional', batch_size=sampling_batch_size)
    z_train = evaluation.z_train
    z_test = evaluation.z_test
    zhat = evaluation.compute_z_gen(xhat)

    print('evaluation for unconditional sampling...')
    wandb.log({'FID': evaluation.fid_score(z_test, zhat)})
    evaluation.log_visual_inspection(evaluation.X_train, xhat, 'X_train vs Xhat')
    evaluation.log_visual_inspection(evaluation.X_test, xhat, 'X_test vs Xhat')
    evaluation.log_visual_inspection(evaluation.X_train, evaluation.X_test, 'X_train vs X_test')
    evaluation.log_pca([z_train], ['Z_train'])
    evaluation.log_pca([z_test], ['Z_test'])
    evaluation.log_pca([zhat], ['Zhat'])
    evaluation.log_pca([z_train, zhat], ['Z_train', 'Zhat'])
    evaluation.log_pca([z_test, zhat], ['Z_test', 'Zhat'])
    evaluation.log_pca([z_train, z_test], ['Z_train', 'Z_test'])
    if evaluation.has_reconstruction_model:
        z_rec_train = evaluation.compute_z_rec('train')
        z_rec_test = evaluation.compute_z_rec('test')
        evaluation.log_pca([z_train, z_rec_train], ['Z_train', 'Z_rec_train'])
        evaluation.log_pca([z_test, z_rec_test], ['Z_test', 'Z_rec_test'])

    mdd, acd, sd, kd = evaluation.stat_metrics(evaluation.X_test, xhat)
    wandb.log({'MDD': mdd, 'ACD': acd, 'SD': sd, 'KD': kd})
    wandb.log({
        'MDD_ratio': mdd / (baseline_metrics['MDD'] + 1e-8),
        'ACD_ratio': acd / (baseline_metrics['ACD'] + 1e-8),
        'SD_ratio': sd / (baseline_metrics['SD'] + 1e-8),
        'KD_ratio': kd / (baseline_metrics['KD'] + 1e-8),
    })

    print('evaluation for class-conditional sampling...')
    n_plot_samples_per_class = 100
    alpha = 0.1
    ylim = (-5, 5)
    n_rows = int(np.ceil(np.sqrt(n_classes)))
    fig1, axes1 = plt.subplots(n_rows, n_rows, figsize=(4 * n_rows, 2 * n_rows))
    fig2, axes2 = plt.subplots(n_rows, n_rows, figsize=(4 * n_rows, 2 * n_rows))
    fig1.suptitle('X_test_c')
    fig2.suptitle(f'Xhat_c ({generator})')
    axes1 = np.atleast_1d(axes1).flatten()
    axes2 = np.atleast_1d(axes2).flatten()
    n_cls_samples, cfids = [], []

    for cls_idx in range(n_classes):
        (_, _, xhat_c), _ = evaluation.sample(n_plot_samples_per_class, kind='conditional', class_index=cls_idx, batch_size=sampling_batch_size)
        cls_sample_ind = (evaluation.Y_test[:] == cls_idx)
        n_cls_samples.append(cls_sample_ind.astype(float).sum())
        try:
            z_test_c = evaluation.compute_z_gen(torch.from_numpy(evaluation.X_test[cls_sample_ind]))
            zhat_c = evaluation.compute_z_gen(xhat_c)
        except ValueError as e:
            print(f'No sample generated for class {cls_idx}: {e}')
            continue
        cfid = evaluation.fid_score(z_test_c, zhat_c)
        cfids.append(cfid)
        wandb.log({f'cFID-cls_{cls_idx}': cfid})
        evaluation.log_pca([z_test_c, zhat_c], [f'Z_test_c{cls_idx}', f'Zhat_c{cls_idx}'])
        mdd_c, acd_c, sd_c, kd_c = evaluation.stat_metrics(torch.from_numpy(evaluation.X_test[cls_sample_ind]), xhat_c)
        wandb.log({f'MDD-cls_{cls_idx}': mdd_c, f'ACD-cls_{cls_idx}': acd_c, f'SD-cls_{cls_idx}': sd_c, f'KD-cls_{cls_idx}': kd_c})
        X_test_c = evaluation.X_test[cls_sample_ind]
        sample_ind = np.random.randint(0, X_test_c.shape[0], n_plot_samples_per_class)
        axes1[cls_idx].plot(X_test_c[sample_ind, 0, :].T, alpha=alpha, color='C0')
        axes1[cls_idx].set_title(f'cls_idx:{cls_idx}')
        axes1[cls_idx].set_ylim(*ylim)
        sample_ind = np.random.randint(0, xhat_c.shape[0], n_plot_samples_per_class)
        axes2[cls_idx].plot(xhat_c[sample_ind, 0, :].T, alpha=alpha, color='C0')
        axes2[cls_idx].set_title(f'cls_idx:{cls_idx}')
        axes2[cls_idx].set_ylim(*ylim)

    if use_custom_dataset and 'svri' in dataset_name.lower():
        print('\n--- SVRI Consistency Evaluation ---')
        cm, acc = evaluate_conditional_generation_consistency(
            evaluation,
            config,
            n_classes=n_classes,
            n_samples_per_class=100,
            sampling_batch_size=sampling_batch_size,
        )
        print(f'SVRI Consistency Accuracy: {acc * 100:.2f}%')
        print(f'SVRI Consistency Confusion Matrix:\n{cm}\n')
        if use_ridge_plots:
            print('\n--- SVRI Ridge Visualization ---')
            cm_ridge, acc_ridge = evaluate_conditional_generation_consistency_ridge(
                evaluation,
                config,
                n_classes=n_classes,
                n_samples_per_class=100,
                sampling_batch_size=sampling_batch_size,
            )
            print(f'SVRI Ridge Accuracy: {acc_ridge * 100:.2f}%')
            print(f'SVRI Ridge Confusion Matrix:\n{cm_ridge}\n')

    fig1.tight_layout()
    fig2.tight_layout()
    wandb.log({'X_test_c': wandb.Image(fig1), 'Xhat_c': wandb.Image(fig2)})
    cfid_mean = np.mean(cfids) if cfids else float('nan')
    wandb.log({'cFID': cfid_mean, 'cFID_ratio': cfid_mean / (baseline_metrics['cFID'] + 1e-8)})
    plt.close(fig1)
    plt.close(fig2)

    fig, ax = plt.subplots()
    ax.bar(range(len(cfids)), cfids)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('FID per class')
    wandb.log({'cFID_bar': wandb.Image(fig)})
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(range(n_classes), n_cls_samples)
    ax.set_xlabel('Class Index')
    ax.set_ylabel('num samples per class')
    wandb.log({'n_cls_samples': wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()


if __name__ == '__main__':
    args = load_args()
    config = load_yaml_param_settings(args.config)
    dataset_names = args.dataset_names if args.dataset_names else pd.read_csv(get_root_dir().joinpath('datasets', 'DataSummary_UCR.csv'))['Name'].tolist()
    for dataset_name in dataset_names:
        batch_size = config['evaluation']['batch_size']
        if not args.use_custom_dataset:
            dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
            train_loader = build_data_pipeline(batch_size, dataset_importer, config, 'train')
        else:
            dataset_importer = DatasetImporterCustomTest(dataset_name, **config['dataset'])
            train_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'train')
        evaluate(
            config,
            dataset_name,
            train_loader,
            args.gpu_device_ind,
            args.use_neural_mapper,
            args.feature_extractor_type,
            args.use_custom_dataset,
            args.sampling_batch_size,
            args.generator,
            args.generator_ckpt,
            args.sampling_steps,
            use_ridge_plots=args.use_ridge_plots,
        )
        torch.cuda.empty_cache()
