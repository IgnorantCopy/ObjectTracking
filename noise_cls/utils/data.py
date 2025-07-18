import os
import re
import glob
import h5py
import numpy as np
from math import log10
from tqdm import tqdm


ABNORMAL_BATCH_ID = [1451, 1452, 1457, 1462, 1467, 1469, 1473, 1478, 1484, 1487, 1488, 1490, 1494, 1496, 1497, 1500]


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


def db(real, imag, eps=1e-10):
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


def load_image(path):
    data = read_mat(path)
    rd_matrix = data['rd_matrix']
    velocity_axis = data['velocity_axis'].reshape(-1)
    velocity_mask = np.abs(velocity_axis) < 56
    velocity_axis = velocity_axis[velocity_mask]
    rd_matrix = rd_matrix[:, velocity_mask]
    value = np.zeros_like(rd_matrix, dtype=np.float64)
    for i in range(len(rd_matrix)):
        for j in range(len(rd_matrix[i])):
            real = rd_matrix[i][j][0]
            imag = rd_matrix[i][j][1]
            value[i][j] = db(real, imag)
    velocity_index = np.where(velocity_axis == 0)[0][0]
    offset = np.random.choice([0, 0, 1, -1])
    noise = value[:, velocity_index + offset]
    other = value[:, 0]
    return noise, other


def main(data_root: str):
    save_root = os.path.join(data_root, "../noise_cls")
    os.makedirs(save_root, exist_ok=True)
    label_dirs = glob.glob(f"{data_root}/Label_*")
    for label_dir in label_dirs:
        label = int(re.match(r".*Label_(\d+).*", label_dir).group(1))
        if label not in [1, 2, 3, 5]:
            continue
        batch_dirs = glob.glob(f"{label_dir}/Batch_*")
        for batch_dir in tqdm(batch_dirs, desc=f"Label {label}"):
            batch = int(re.match(r".*Batch_(\d+).*", batch_dir).group(1))
            if batch in ABNORMAL_BATCH_ID:
                continue
            frame_dirs = glob.glob(f"{batch_dir}/Frame_*")
            for frame_dir in frame_dirs:
                frame = int(re.match(r".*Frame_(\d+).*", frame_dir).group(1))
                noise, other = load_image(f"{frame_dir}/MTD_result.mat")
                save_dir = os.path.join(save_root, f"Batch_{batch}/Frame_{frame}")
                os.makedirs(save_dir, exist_ok=True)
                noise_path = os.path.join(save_dir, "noise.npy")
                if not os.path.exists(noise_path):
                    np.save(os.path.join(save_dir, "noise.npy"), noise)
                other_path = os.path.join(save_dir, "other.npy")
                if not os.path.exists(other_path):
                    np.save(os.path.join(save_dir, "other.npy"), other)


if __name__ == '__main__':
    main(r"D:\DataSets\挑战杯_揭榜挂帅_CQ-08赛题_数据集\MTD_Results")