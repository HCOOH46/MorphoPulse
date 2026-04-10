"""
FID, IS
"""
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm

# from supervised_FCN_2.example_pretrained_model_loading import load_pretrained_FCN
# from supervised_FCN_2.example_compute_FID import calculate_fid
# from supervised_FCN_2.example_compute_IS import calculate_inception_score
from generators.sample import unconditional_sample, conditional_sample

from evaluation.rocket_functions import generate_kernels, apply_kernels
from preprocessing.preprocess_ucr import DatasetImporterUCR, DatasetImporterCustom, DatasetImporterCustomTest
from utils import freeze, remove_outliers
from evaluation.stat_metrics import marginal_distribution_difference, auto_correlation_difference, skewness_difference, kurtosis_difference

from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from numpy.random import random
from scipy.linalg import sqrtm
def calculate_inception_score(P_yx, n_split: int = 10, shuffle: bool = True, eps: float = 1e-16):
    """
    P_yx: (batch_size dim)
    """
    if shuffle:
        np.random.shuffle(P_yx)  # in-place

    scores = list()
    n_part = int(np.floor(P_yx.shape[0] / n_split))
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = P_yx[ix_start:ix_end]

        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)

        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))

        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)

        # average over images
        avg_kl_d = mean(sum_kl_d)

        # undo the log
        is_score = exp(avg_kl_d)

        # store
        scores.append(is_score)

    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

def calculate_fid(z1, z2):
    """
    :param z1: representation after the last pooling layer (e.g., GAP)
    :param z2: representation after the last pooling layer (e.g., GAP)
    :return: FID score
    """
    # calculate mean and covariance statistics
    mu1, sigma1 = z1.mean(axis=0), cov(z1, rowvar=False)
    mu2, sigma2 = z2.mean(axis=0), cov(z2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = ((mu1 - mu2) ** 2.0).sum()

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

@torch.no_grad()
def sample(batch_size:int, maskgit, device, n_samples: int, kind: str, class_index:Union[None,int]):
    assert kind in ['unconditional', 'conditional']

    # sampling
    if kind == 'unconditional':
        x_new_l, x_new_h, x_new = unconditional_sample(maskgit, n_samples, device, batch_size=batch_size)  # (b c l); b=n_samples, c=1 (univariate)
    elif kind == 'conditional':
        x_new_l, x_new_h, x_new = conditional_sample(maskgit, n_samples, device, class_index, batch_size)  # (b c l); b=n_samples, c=1 (univariate)
    else:
        raise ValueError

    return x_new_l, x_new_h, x_new


class Metrics(object):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, 
                 config:dict,
                 dataset_name: str, 
                 n_classes:int,
                 feature_extractor_type:str, 
                 rocket_num_kernels:int=1000,
                 batch_size: int=32,
                 use_custom_dataset:bool=False,
                 device='cuda',
                 test=False
                 ):
        self.dataset_name = dataset_name
        self.feature_extractor_type = feature_extractor_type
        self.batch_size = batch_size
        self.device = device

        # load the numpy matrix of the test samples
        dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset']) if not use_custom_dataset else (DatasetImporterCustomTest(dataset_name) if test else DatasetImporterCustom(dataset_name, **config['dataset']))
        self.X_train = dataset_importer.X_train
        self.X_train = dataset_importer.X_train  # (b 1 l)
        self.X_test = dataset_importer.X_test  # (b 1 l)
        self.n_classes = n_classes
        
        # load a model
        if self.feature_extractor_type == 'supervised_fcn':
            self.fcn = load_pretrained_FCN(dataset_name)
            freeze(self.fcn)
            self.fcn.eval()
        elif self.feature_extractor_type == 'papagei':
            from outside_call import get_papagei, get_param
            fcn0 = get_papagei()
            self.fcn = get_param(fcn0)
            self.fcn.to(device)
            freeze(self.fcn)
            self.fcn.eval()
        elif self.feature_extractor_type == 'PM':
            from outside_call import get_papagei, get_param
            fcn0 = get_papagei("Mamba1")
            self.fcn = get_param(fcn0, 'Mamba1')
            self.fcn.to(device)
            self.fcn.eval()
        
        input_length = self.X_train.shape[-1]
        self.rocket_kernels = generate_kernels(input_length, num_kernels=rocket_num_kernels)
        
        # compute z_train, z_test
        self.z_train = self.compute_z(self.X_train)  # (b d)
        self.z_test = self.compute_z(self.X_test)  # (b d)

    @torch.no_grad()
    def sample(self, maskgit, device, n_samples: int, kind: str, class_index:Union[None,int]):
        return sample(self.batch_size, maskgit, device, n_samples, kind, class_index)
        
    def extract_feature_representations(self, x:np.ndarray, feature_extractor_type:str=None):
        """
        x: (b 1 l)
        """
        feature_extractor_type = feature_extractor_type if not isinstance(feature_extractor_type, type(None)) else self.feature_extractor_type
        if feature_extractor_type == 'papagei':
            z = self.fcn(torch.from_numpy(x).float().to(self.device))[0].cpu().detach().numpy()  # (b d)
        elif feature_extractor_type == 'PM':
            z = self.fcn(torch.from_numpy(x).float().to(self.device))[0].cpu().detach().numpy()  # (b d)
        elif feature_extractor_type == 'supervised_fcn':
            device = next(self.fcn.parameters()).device
            z = self.fcn(torch.from_numpy(x).float().to(self.device), return_feature_vector=True).cpu().detach().numpy()  # (b d)
        elif feature_extractor_type == 'rocket':
            x = x[:,0,:].astype(float)  # (b l)
            z = apply_kernels(x, self.rocket_kernels)  # (b d)
            z = F.normalize(torch.from_numpy(z), p=2, dim=1).numpy()
        else:
            raise ValueError
        return z
    
    def compute_z_stat(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z = self.extract_feature_representations(x[s])
            zs.append(z)
        zs = np.concatenate(zs, axis=0)
        z_mu, z_std = np.mean(zs, axis=0)[None,:], np.std(zs, axis=0)[None,:]  # (1 d), (1 d)
        return z_mu, z_std
    
    def compute_z(self, x: np.ndarray, max_samples=40000) -> np.ndarray:
        # n_samples = x.shape[0]
        # n_iters = n_samples // self.batch_size
        # if n_samples % self.batch_size > 0:
        #     n_iters += 1

        # # get feature vectors
        # zs = []
        # for i in range(n_iters):
        #     s = slice(i * self.batch_size, (i + 1) * self.batch_size)
        #     z = self.extract_feature_representations(x[s])
        #     zs.append(z)
        # zs = np.concatenate(zs, axis=0)
        # return zs
        # 只随机采样 max_samples 个样本
        if x.shape[0] > max_samples:
            idx = np.random.choice(x.shape[0], max_samples, replace=False)
            x = x[idx]
        n_samples = x.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        zs = []
        for i in tqdm(range(n_iters), desc="Computing feature representations", total=n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z = self.extract_feature_representations(x[s])
            zs.append(z)
        zs = np.concatenate(zs, axis=0)
        return zs
    
    def z_gen_fn(self, x_gen: np.ndarray):
        z_gen = self.compute_z(x_gen)
        return z_gen

    # def fid_score(self, x_gen: np.ndarray):
    #     z_gen = self.z_gen_fn(x_gen)
    #     z_gen = remove_outliers(z_gen)

    #     fid_train_gen = calculate_fid(self.z_train, z_gen)
    #     fid_test_gen = calculate_fid(self.z_test, z_gen)
    #     return fid_train_gen, fid_test_gen
    
    def fid_score(self, z1:np.ndarray, z2:np.ndarray) -> int:
        z1, z2 = remove_outliers(z1), remove_outliers(z2)
        fid = calculate_fid(z1, z2)
        return fid
    
    def inception_score(self, x_gen: np.ndarray):
        device = next(self.fcn.parameters()).device
        n_samples = x_gen.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get the softmax distribution from `x_gen`
        p_yx_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            p_yx_g = self.fcn(torch.from_numpy(x_gen[s]).float().to(device))  # p(y|x)
            p_yx_g = torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy()

            p_yx_gen.append(p_yx_g)
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)

        IS_mean, IS_std = calculate_inception_score(p_yx_gen, n_split=5)
        return IS_mean, IS_std

    def stat_metrics(self, x_real:np.ndarray, x_gen:np.ndarray) -> Tuple[float, float, float, float]:
        """
        computes the statistical metrices introduced in the paper, [Ang, Yihao, et al. "Tsgbench: Time series generation benchmark." arXiv preprint arXiv:2309.03755 (2023).]

        x_real: (batch 1 length)
        x_gen: (batch 1 length)
        """
        mdd = marginal_distribution_difference(x_real, x_gen)
        acd = auto_correlation_difference(x_real, x_gen)
        sd = skewness_difference(x_real, x_gen)
        kd = kurtosis_difference(x_real, x_gen)
        return mdd, acd, sd, kd
