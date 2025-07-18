import os
import glob
import random
import numpy as np
from torch.utils.data import Dataset


NUM_CLASSES = 2
label_map = {"noise": 0, "other": 1}


def split_train_val(data_root: str, val_ratio=0.2, shuffle=True):
    batch_dirs = glob.glob(f"{data_root}/Batch_*")
    batch_dirs.sort()
    total = 0
    data_paths = [[] for _ in range(NUM_CLASSES)]
    for batch_dir in batch_dirs:
        frame_dirs = glob.glob(f"{batch_dir}/Frame_*")
        frame_dirs.sort()
        for frame_dir in frame_dirs:
            data_paths[0].append(os.path.join(frame_dir, "noise.npy"))
            data_paths[1].append(os.path.join(frame_dir, "other.npy"))
            total += 1
    assert total == len(data_paths[0]) == len(data_paths[1]), \
        f"Number of data paths is not equal: {total} vs {len(data_paths[0])}, {len(data_paths[1])}"
    train_nums = int(total * (1 - val_ratio))
    val_nums = total - train_nums
    train_paths, val_paths = [], []
    for i in range(NUM_CLASSES):
        data_path = data_paths[i]
        if shuffle:
            random.shuffle(data_path)
        train_paths.extend(data_path[:train_nums])
        val_paths.extend(data_path[train_nums:train_nums + val_nums])
    np.random.shuffle(train_paths)
    np.random.shuffle(val_paths)
    return train_paths, val_paths


class RDCol(Dataset):
    def __init__(self, data_paths: list):
        super().__init__()
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, item):
        data_path = self.data_paths[item]
        data = np.load(data_path).astype(np.float32)
        label = label_map[os.path.basename(data_path).split('.')[0]]
        return data, label