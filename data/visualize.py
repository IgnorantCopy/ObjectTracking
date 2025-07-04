import os
import h5py
import glob
import numpy as np
from math import sqrt, log10
from matplotlib import pyplot as plt
from frame_wise.utils.dataset import split_scientific_str


def read_matlab_v73_file(label, batch, frame):
    file_path = f"E:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集/MTD_Results/Label_{label}/Batch_{batch}/Frame_{frame}/MTD_result.mat"
    with h5py.File(file_path, 'r') as f:
        var_names = list(f.keys())
        data = {}

        for var_name in var_names:
            # HDF5 中的数据是按列优先存储的，而 NumPy 是按行优先，因此需要转置
            if isinstance(f[var_name], h5py.Dataset):
                data[var_name] = np.array(f[var_name]).T

        return data


def check_path(path):
    path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_rd_matrix(data, save_path=None, eps=1e-6, verbose=False):
    """可视化 RD 矩阵"""
    rd_matrix = data['rd_matrix']
    value = np.zeros_like(rd_matrix, dtype=np.float64)
    for i in range(len(rd_matrix)):
        for j in range(len(rd_matrix[i])):
            real = rd_matrix[i][j][0]
            real_mantissa, real_exponent = split_scientific_str(real)
            imag = rd_matrix[i][j][1]
            imag_mantissa, imag_exponent = split_scientific_str(imag)
            if real_exponent > imag_exponent:
                real_mantissa *= 10 ** (real_exponent - imag_exponent)
                value[i][j] = 10 * (log10(real_mantissa ** 2 + imag_mantissa ** 2 + eps) +  2 * imag_exponent)
            elif real_exponent < imag_exponent:
                imag_mantissa *= 10 ** (imag_exponent - real_exponent)
                value[i][j] = 20 * (log10(sqrt(real_mantissa ** 2 + imag_mantissa ** 2 + eps)) + real_exponent)
            else:
                value[i][j] = 20 * (log10(sqrt(real_mantissa ** 2 + imag_mantissa ** 2 + eps)) + real_exponent)

    range_axis = data['range_axis']
    velocity_axis = data['velocity_axis']

    if verbose:
        print(f"RD矩阵形状: {rd_matrix.shape}")
        print(f"距离轴长度: {len(range_axis)}")
        print(f"速度轴长度: {len(velocity_axis)}")

    plt.figure(figsize=(10, 8))

    X, Y = np.meshgrid(velocity_axis, range_axis)
    plt.pcolormesh(X, Y, value, shading='auto', cmap='jet')
    plt.xlim(-30, 30)
    plt.colorbar(label='幅度')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('距离 (m)')
    plt.title('距离-多普勒图 (RD Map)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        check_path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



if __name__ == '__main__':
    data_root = "E:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集/MTD_Results"
    label_dirs = glob.glob(f"{data_root}/Label_*")
    for label_dir in label_dirs:
        label = int(label_dir.split("_")[-1])
        batch_dirs = glob.glob(f"{label_dir}/Batch_*")
        for batch_dir in batch_dirs:
            batch = int(batch_dir.split("_")[-1])
            frame_dirs = glob.glob(f"{batch_dir}/Frame_*")
            for frame_dir in frame_dirs:
                frame = int(frame_dir.split("_")[-1])
                save_path = f"{frame_dir}/rd_map.png"
                if os.path.exists(save_path):
                    continue
                data = read_matlab_v73_file(label, batch, frame)
                visualize_rd_matrix(data, save_path=save_path)
                print(f"label: {label}, batch: {batch}, frame: {frame} saved!")
    print("All done!")