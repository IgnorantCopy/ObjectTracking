import os
import re
import h5py
import glob
import random
import numpy as np
from math import log10
from torch.utils.data import Dataset


def split_train_val(data_root: str, val_ratio=0.2, shuffle=True):
    label_dirs = glob.glob(f"{data_root}/Label_*")
    label_dirs.sort()
    label_nums = []
    data_paths = []
    for label_dir in label_dirs:
        batch_dirs = glob.glob(f"{label_dir}/Batch_*")
        batch_dirs.sort()
        label_num = 0
        label_paths = []
        for batch_dir in batch_dirs:
            frame_dirs = glob.glob(f"{batch_dir}/Frame_*")
            frame_dirs.sort()
            label_num += len(frame_dirs)
            label_paths.extend(frame_dirs)
        label_nums.append(label_num)
        data_paths.append(label_paths)
    train_nums = [int(num * (1 - val_ratio)) for num in label_nums]
    val_nums = [num - train_num for num, train_num in zip(label_nums, train_nums)]
    train_paths, val_paths = [], []
    for i, data_path in enumerate(data_paths):
        if shuffle:
            random.shuffle(data_path)
        train_paths.extend(data_path[:train_nums[i]])
        val_paths.extend(data_path[train_nums[i]:val_nums[i]])
    return train_paths, val_paths


def split_scientific_str(num, precision=10):
    """使用字符串格式化将一个数分成科学计数法的尾数和指数两部分"""
    scientific_str = f"{num:.{precision}e}"
    match = re.match(r'([+-]?\d+\.\d+)e([+-]?\d+)', scientific_str)
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa, exponent
    return num, 0


def read_mat(path):
    with h5py.File(path, 'r') as f:
        var_names = list(f.keys())
        data = {}
        for var_name in var_names:
            if isinstance(f[var_name], h5py.Dataset):
                data[var_name] = np.array(f[var_name]).T
    return data


class RDMap(Dataset):
    def __init__(self, data_paths: list, transform=None):
        super().__init__()
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, item):
        data_path = self.data_paths[item]
        image_path = os.path.join(data_path, "MTD_result.mat")
        image = self._load_image(image_path)
        cls = int(re.match(r".*Label_(\d+).*", data_path).group(1))
        if self.transform:
            image = self.transform(image)
        return image, cls - 1

    @staticmethod
    def _db(real, imag, eps=1e-10):
        real_mantissa, real_exponent = split_scientific_str(real)
        imag_mantissa, imag_exponent = split_scientific_str(imag)
        if real_exponent > imag_exponent:
            real_mantissa *= 10 ** (real_exponent - imag_exponent)
            return 10 * (log10(real_mantissa ** 2 + imag_mantissa ** 2 + eps) + 2 * imag_exponent)
        elif real_exponent < imag_exponent:
            imag_mantissa *= 10 ** (imag_exponent - real_exponent)
            return 10 * (log10(real_mantissa ** 2 + imag_mantissa ** 2 + eps) + 2 * real_exponent)
        else:
            return 10 * (log10(real_mantissa ** 2 + imag_mantissa ** 2 + eps) + 2 * real_exponent)

    def _load_image(self, path):
        data = read_mat(path)
        rd_matrix = data['rd_matrix']
        velocity_axis = data['velocity_axis']
        velocity_mask = np.reshape(np.abs(velocity_axis) < 56, -1)
        rd_matrix = rd_matrix[:, velocity_mask]
        value = np.zeros_like(rd_matrix, dtype=np.float64)
        for i in range(len(rd_matrix)):
            for j in range(len(rd_matrix[i])):
                real = rd_matrix[i][j][0]
                imag = rd_matrix[i][j][1]
                value[i][j] = self._db(real, imag)
        return value[:, :, None]
