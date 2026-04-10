"""
`Dataset` (pytorch) class is defined.
"""
from typing import Union
import math

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor

from utils import get_root_dir, download_ucr_datasets
from project_paths import DATA_ROOT


class DatasetImporterUCR(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """
    def __init__(self, dataset_name: str, data_scaling: bool, **kwargs):
        """
        :param dataset_name: e.g., "ElectricDevices"
        :param data_scaling
        """
        download_ucr_datasets()
        # self.data_root = get_root_dir().joinpath("datasets", "UCRArchive_2018", dataset_name)
        self.data_root = get_root_dir().joinpath("datasets", "UCRArchive_2018_resplit", dataset_name)

        # fetch an entire dataset
        df_train = pd.read_csv(self.data_root.joinpath(f"{dataset_name}_TRAIN.tsv"), sep='\t', header=None)
        df_test = pd.read_csv(self.data_root.joinpath(f"{dataset_name}_TEST.tsv"), sep='\t', header=None)

        self.X_train, self.X_test = df_train.iloc[:, 1:].values[:, np.newaxis, :], df_test.iloc[:, 1:].values[:, np.newaxis, :]  # (b 1 l)
        self.Y_train, self.Y_test = df_train.iloc[:, [0]].values[:, np.newaxis, :], df_test.iloc[:, [0]].values[:, np.newaxis, :]  # (b 1 l)

        le = LabelEncoder()
        self.Y_train = le.fit_transform(self.Y_train.ravel())[:, None]
        self.Y_test = le.transform(self.Y_test.ravel())[:, None]

        # if data_scaling:
        #     # following [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30]
        #     mean = np.nanmean(self.X_train)
        #     var = np.nanvar(self.X_train)
        #     self.X_train = (self.X_train - mean) / math.sqrt(var)
        #     self.X_test = (self.X_test - mean) / math.sqrt(var)
        self.mean, self.std = 1., 1.
        if data_scaling:
            self.mean = np.nanmean(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.std = np.nanstd(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.X_train = (self.X_train - self.mean) / self.std  # (b c l)
            self.X_test = (self.X_test - self.mean) / self.std  # (b c l)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train.reshape(-1)))
        print("# unique labels (test):", np.unique(self.Y_test.reshape(-1)))


class UCRDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporterUCR,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        self.kind = kind

        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self._len = self.X.shape[0]
    
    def __getitem__(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        return x, y

    def __len__(self):
        return self._len



# class DatasetImporterCustom(object):
#     def __init__(self, data_scaling:bool=True, **kwargs):
#         # training and test datasets
#         # typically, you'd load the data, for example, using pandas
#         self.X_train, self.Y_train = np.random.rand(800, 1, 100), np.random.randint(0, 2, size=(800,1))  # X:(n_training_samples, 1, ts_length); 1 denotes a univariate time series; 2 classes in this example
#         self.X_test, self.Y_test = np.random.rand(200, 1, 100), np.random.randint(0, 2, size=(200,1))  # (n_test_samples, 1, ts_length)

#         if data_scaling:
#             # following [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30]
#             mean = np.nanmean(self.X_train)
#             var = np.nanvar(self.X_train)
#             self.X_train = (self.X_train - mean) / math.sqrt(var)
#             self.X_test = (self.X_test - mean) / math.sqrt(var)

#         np.nan_to_num(self.X_train, copy=False)
#         np.nan_to_num(self.X_test, copy=False)
        
class DatasetImporterCustom(object):
    def __init__(self, dataset_name, data_scaling:bool=True, base_path=None, conditioned=False, **kwargs):
        if base_path is None:
            base_path = DATA_ROOT / 'generation'
        path = os.path.join(base_path, dataset_name)
        assert os.path.exists(path), f'{path} does not exist.'
        # training and test datasets
        # typically, you'd load the data, for example, using pandas
        
        # Parameters
        # n_train_samples = 8000
        # n_test_samples = 2000
        # ts_length = 100
        # n_channels = 2
        
    # with np.load(train_path, allow_pickle=False) as npz:
    #     self.X_train = npz['segments'] # (b, 1, l)
    #     self.Y_train = npz['labels'] # (b, 1)
        
    # with np.load(test_path, allow_pickle=False) as npz:
    #     self.X_test = npz['segments']
    #     self.Y_test = npz['labels']
        print("normal loader")
        print('\033[92m' + "Loading data from npy files..." + '\033[0m')
        self.X_train = np.load(os.path.join(path, 'train_segments.npy'), mmap_mode='r')
        self.Y_train = np.load(os.path.join(path, 'train_labels.npy'), mmap_mode='r')
        self.X_test = np.load(os.path.join(path, 'test_segments.npy'), mmap_mode='r')
        self.Y_test = np.load(os.path.join(path, 'test_labels.npy'), mmap_mode='r')
        print('\033[92m' + "Data loaded." + '\033[0m')
        
        if not conditioned:
            print("\033[33mUnconditional training: labels set to 0.\033[0m")
            self.Y_train = np.zeros_like(self.Y_train)
            self.Y_test = np.zeros_like(self.Y_test)
            
        
        # # Generate sine time series with random phases for training data
        # self.X_train = 2*np.sin(np.linspace(0, 2 * np.pi, ts_length) + np.random.rand(n_train_samples, n_channels, 1) * 2 * np.pi)  # (n_training_samples, n_channels, length) = (b c l)
        # self.Y_train = np.random.randint(0, 2, size=(n_train_samples, 1))  # (n_training_samples 1)

        # # Generate sine time series with random phases for test data
        # self.X_test = 2*np.sin(np.linspace(0, 2 * np.pi, ts_length) + np.random.rand(n_test_samples, n_channels, 1) * 2 * np.pi)  # (n_test_samples, n_channels, length) = (b c l)
        # self.Y_test = np.random.randint(0, 2, size=(n_test_samples, 1))  # (n_test_samples 1)

        self.mean, self.std = 1., 1.
        if data_scaling:
            # Z-score normalization
            self.mean = np.nanmean(np.asarray(self.X_train), axis=(0, 2))[None,:,None]  # (1 c 1)
            self.std = np.nanstd(np.asarray(self.X_train), axis=(0, 2))[None,:,None]  # (1 c 1)
    #     self.X_train = (self.X_train - self.mean) / self.std  # (b c l)
    #     self.X_test = (self.X_test - self.mean) / self.std  # (b c l)

    # np.nan_to_num(self.X_train, copy=False)
    # np.nan_to_num(self.X_test, copy=False)
    
# 多进程采样
# def sample_idx(arr, max_samples):
#     if arr.shape[0] > max_samples:
#         idx = np.random.choice(arr.shape[0], max_samples, replace=False)
#         return arr[idx]
#     return arr
def stratified_sample(X, Y, max_samples):
    """
    按类别分布进行分层采样，确保各类别都有足够的样本
    """
    # 统计各类别的分布
    unique_labels, counts = np.unique(Y.flatten(), return_counts=True)
    total_samples = len(Y)
    
    print(f"原始数据类别分布: {dict(zip(unique_labels, counts))}")
    
    if total_samples <= max_samples:
        return X, Y
    
    # 计算各类别应该采样的数量（按比例）
    sample_ratios = counts / total_samples
    target_counts = np.round(sample_ratios * max_samples).astype(int)
    
    # 确保每个类别至少有1个样本
    min_samples_per_class = 1
    target_counts = np.maximum(target_counts, min_samples_per_class)
    
    # 如果总数超过max_samples，按比例缩减
    if target_counts.sum() > max_samples:
        # 优先保证每个类别至少有1个样本，然后按比例分配剩余样本
        remaining_samples = max_samples - len(unique_labels)
        if remaining_samples > 0:
            # 重新计算比例，排除已分配的最小样本
            remaining_counts = counts - min_samples_per_class
            remaining_ratios = remaining_counts / remaining_counts.sum()
            additional_counts = np.floor(remaining_ratios * remaining_samples).astype(int)
            target_counts = min_samples_per_class + additional_counts
        else:
            target_counts = np.ones(len(unique_labels), dtype=int)
    
    print(f"目标采样数量: {dict(zip(unique_labels, target_counts))}")
    
    # 为每个类别采样
    sampled_indices = []
    for label, target_count in zip(unique_labels, target_counts):
        # 找到该类别的所有样本索引
        label_indices = np.where(Y.flatten() == label)[0]
        
        if len(label_indices) <= target_count:
            # 如果该类别样本数不足，全部采样
            sampled_indices.extend(label_indices)
        else:
            # 随机采样指定数量
            sampled = np.random.choice(label_indices, target_count, replace=False)
            sampled_indices.extend(sampled)
    
    # 转换为numpy数组并排序
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)  # 打乱顺序
    
    # 验证采样结果
    final_labels, final_counts = np.unique(Y[sampled_indices].flatten(), return_counts=True)
    print(f"实际采样结果: {dict(zip(final_labels, final_counts))}")
    
    return X[sampled_indices], Y[sampled_indices]


def calc_mean_std(arr):
    arr = np.asarray(arr)
    mean = np.nanmean(arr, axis=(0, 2))[None, :, None]
    std = np.nanstd(arr, axis=(0, 2))[None, :, None]
    return mean, std

def normalize(arr, mean, std):
    arr = (arr - mean) / std
    np.nan_to_num(arr, copy=False)
    return arr

def analyze_class_distribution(Y, dataset_type=""):
    """分析并打印类别分布"""
    unique_labels, counts = np.unique(Y.flatten(), return_counts=True)
    total = len(Y)
    
    print(f"\n{dataset_type} 类别分布分析:")
    print(f"总样本数: {total}")
    print(f"类别数量: {len(unique_labels)}")
    print("各类别分布:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / total) * 100
        print(f"  类别 {label}: {count} 样本 ({percentage:.2f}%)")
    
    return unique_labels, counts


class DatasetImporterCustomTest(object):
    def __init__(self, dataset_name, data_scaling=True, base_path=None, max_samples=20000, **kwargs):
        if base_path is None:
            base_path = DATA_ROOT / 'generation'
        path = os.path.join(base_path, dataset_name)
        assert os.path.exists(path), f'{path} does not exist.'
        self.X_train = np.load(os.path.join(path, 'train_segments.npy'), mmap_mode='r')
        self.Y_train = np.load(os.path.join(path, 'train_labels.npy'), mmap_mode='r')
        self.X_test = np.load(os.path.join(path, 'test_segments.npy'), mmap_mode='r')
        self.Y_test = np.load(os.path.join(path, 'test_labels.npy'), mmap_mode='r')
        try:
            assert len(np.unique(self.Y_train)) == len(np.unique(self.Y_test))
        except AssertionError:
            print(f'\033[32m{len(np.unique(self.Y_train))} classes in train, {len(np.unique(self.Y_test))} classes in test.\033[0m')
        
        # 分析原始数据的类别分布
        train_labels, train_counts = analyze_class_distribution(self.Y_train, "训练集")
        test_labels, test_counts = analyze_class_distribution(self.Y_test, "测试集")
        
        # 检查训练集和测试集的类别是否一致
        if not np.array_equal(sorted(train_labels), sorted(test_labels)):
            print(f'\033[33m警告: 训练集和测试集的类别不完全一致!\033[0m')
            print(f'训练集类别: {sorted(train_labels)}')
            print(f'测试集类别: {sorted(test_labels)}')
        
        # 使用分层采样
        print(f"\n开始分层采样 (最大样本数: {max_samples})...")
        
        print("resampling")
        self.X_train, self.Y_train = stratified_sample(self.X_train, self.Y_train, max_samples)
        self.X_test, self.Y_test = stratified_sample(self.X_test, self.Y_test, max_samples)

        # 多进程归一化
        self.mean, self.std = 1., 1.
        if data_scaling:
            
            print("calculating mean and std")
            self.mean, self.std = calc_mean_std(self.X_train)

            print("normalizing")
            self.X_train = normalize(self.X_train, self.mean, self.std)
            self.X_test = normalize(self.X_test, self.mean, self.std)
                

class CustomDataset(Dataset):
    def __init__(self, kind: str, dataset_importer:DatasetImporterCustom, **kwargs):
        """
        :param kind: "train" | "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        kind = kind.lower()
        assert kind in ['train', 'test']
        self.kind = kind
        
        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self.mean, self.std = dataset_importer.mean, dataset_importer.std
        self._len = self.X.shape[0]
    
    def __getitem__(self, idx):
        # Z-score normalization，移到getitem里，节省内存
        x = np.nan_to_num((self.X[idx, ...] - self.mean) / self.std)
        y = self.Y[idx, ...]
        x = np.squeeze(x, axis=1)  # 或 x = x.reshape(channel, length)
        import torch
        return torch.tensor(x).numpy(), torch.tensor(y).numpy()

    def __len__(self):
        return self._len


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    os.chdir("../")

    # data pipeline
    dataset_importer = DatasetImporterUCR("ScreenType", data_scaling=True)
    dataset = UCRDataset("train", dataset_importer)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in data_loader:
        x, y = batch
        break
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)
    print(y.flatten())

    # plot
    n_samples = 10
    c = 0
    fig, axes = plt.subplots(n_samples, 2, figsize=(3.5*2, 1.7*n_samples))
    for i in range(n_samples):
        axes[i,0].plot(x[i, c])
        axes[i,0].set_title(f'class: {y[i,0]}')
        xf = torch.stft(x[[i], c], n_fft=4, hop_length=1, normalized=False)
        print('xf.shape:', xf.shape)
        xf = np.sqrt(xf[0,:,:,0]**2 + xf[0,:,:,1]**2)
        axes[i,1].imshow(xf, aspect='auto')
        axes[i, 1].invert_yaxis()
    plt.tight_layout()
    plt.show()
