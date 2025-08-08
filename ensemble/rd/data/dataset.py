import random
import torch
from torch.utils.data import Dataset

from ensemble.rd.data.rd_preprocess import *


ABNORMAL_BATCH_ID = [1451, 1452, 1457, 1462, 1467, 1469, 1473, 1478, 1484, 1487, 1488, 1490, 1494, 1496, 1497, 1500]


def split_train_val(data_root: str, num_classes, val_ratio=0.2, shuffle=True, seed=42):
    np.random.seed(seed)
    label_nums = [0 for _ in range(num_classes)]
    batch_files_by_cls = [[] for _ in range(num_classes)]
    batch_files = get_batch_file_list(data_root)
    for batch_file in batch_files:
        cls = batch_file.label - 1
        if cls < 0 or cls >= num_classes or batch_file.batch_num in ABNORMAL_BATCH_ID:
            continue
        label_nums[cls] += 1
        batch_files_by_cls[cls].append(batch_file)
    train_nums = [int(num * (1 - val_ratio)) for num in label_nums]
    val_nums = [num - train_num for num, train_num in zip(label_nums, train_nums)]
    train_batch_files, val_batch_files = [], []
    for i, batch_file in enumerate(batch_files_by_cls):
        if shuffle:
            random.shuffle(batch_file)
        train_batch_files.extend(batch_file[:train_nums[i]])
        val_batch_files.extend(batch_file[train_nums[i]:train_nums[i] + val_nums[i]])
    return train_batch_files, val_batch_files


class RDMap(Dataset):
    def __init__(self, batch_files: list[BatchFile], image_transform=None, track_transform=None,
                 image_seq_len=180, track_seq_len=29):
        super().__init__()
        self.batch_files = batch_files
        self.image_transform = image_transform
        self.track_transform = track_transform
        self.image_seq_len = image_seq_len
        self.track_seq_len = track_seq_len

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, item):
        batch_file = self.batch_files[item]
        point_file = batch_file.point_file
        num_points = int(os.path.basename(point_file).split('_')[-1].split('.')[0])
        cls = batch_file.label - 1

        # load rd map
        images, point_index, extra_features, missing_rate = self._load_rd_map(batch_file, num_points)
        if len(images) == 0:
            return batch_file, None, None, None, None, None, cls

        image_mask = np.ones((self.image_seq_len,), dtype=np.int32)
        if images.shape[0] < self.image_seq_len:
            image_mask[images.shape[0]:] = 0
            images = np.concatenate([
                images,
                np.zeros((self.image_seq_len - images.shape[0], *images.shape[1:]))
            ], axis=0)
            point_index = np.concatenate([
                point_index,
                np.array([point_index[-1] for _ in range(self.image_seq_len - point_index.shape[0])])
            ], axis=0)
            extra_features = np.concatenate([
                extra_features,
                np.ones((self.image_seq_len - extra_features.shape[0], extra_features.shape[1])) * np.mean(extra_features)
            ], axis=0)
            missing_rate = np.concatenate([
                missing_rate,
                np.array([missing_rate[-1] for _ in range(self.image_seq_len - missing_rate.shape[0])])
            ], axis=0)
        elif images.shape[0] > self.image_seq_len:
            quantiles = np.linspace(0, 1, self.image_seq_len)
            indices = np.floor(quantiles * (images.shape[0] - 1)).astype(int)
            images = images[indices]
            point_index = point_index[indices]
            extra_features = extra_features[indices]
            missing_rate = missing_rate[indices]
        assert images.shape[0] == self.image_seq_len, f"RD 图数量与预期不符: {images.shape[0]}, {self.image_seq_len}"

        return batch_file, point_index, images, extra_features, missing_rate, image_mask, cls

    def _load_rd_map(self, batch: BatchFile, num_points: int):
        """处理单个批次的数据"""
        # 打开原始数据文件
        frame_count = 0
        rd_matrices = []
        point_index = []
        # 额外特征
        num_none_zero = []
        num_none_zero_row = []
        missing_rate = []
        total = 0
        detected = 0

        with open(batch.raw_file, 'rb') as fid:
            while True:
                params, data = read_raw_data(fid)
                if params is None or data is None:
                    break

                frame_count += 1

                # 跳过没有航迹信息的帧
                if len(params.track_no_info) == 0:
                    continue

                # 添加数据验证
                if len(params.track_no_info) < 4:
                    continue

                # 验证参数有效性
                if params.prt <= 0 or params.prt_num <= 0 or params.freq <= 0:
                    continue

                # MTD处理
                distance_bins = data.shape[0]  # 距离单元数 (31)
                prt_bins = data.shape[1]  # PRT数
                # 生成泰勒窗 - 使用PRT数作为窗长，匹配MATLAB
                mtd_win = signal.windows.taylor(distance_bins, nbar=4, sll=30, norm=False)
                mtd_win_col = mtd_win.reshape(-1, 1)
                # 在距离维度重复窗函数
                coef_mtd_2d = np.repeat(mtd_win_col, prt_bins, axis=1)
                # 加窗处理
                data_windowed = data * coef_mtd_2d
                # FFT处理 - 在PRT维度（轴1）进行FFT
                mtd_result = fftshift(fft(data_windowed, axis=1), axes=1)

                # 计算多普勒速度轴 - 修复溢出问题
                delta_v = C / (2 * params.prt_num * params.prt * params.freq)

                # 检查delta_v是否有效
                if not np.isfinite(delta_v) or delta_v <= 0 or delta_v > 10000:
                    print(f"警告：帧 {frame_count} delta_v异常: {delta_v}, 跳过该帧")
                    continue

                # 修复溢出问题 - 使用更安全的方式
                half_prt = params.prt_num // 2

                # 检查half_prt是否合理
                if half_prt <= 0 or half_prt > 10000:
                    print(f"警告：帧 {frame_count} half_prt异常: {half_prt}, 跳过该帧")
                    continue

                # 使用int32避免溢出
                v_start = -int(half_prt)
                v_end = int(half_prt)
                v_indices = np.arange(v_start, v_end, dtype=np.int32)
                v_axis = v_indices.astype(np.float64) * delta_v

                # 检查v_axis是否有效
                if not np.all(np.isfinite(v_axis)) or len(v_axis) != params.prt_num:
                    print(
                        f"警告：帧 {frame_count} v_axis异常，长度:{len(v_axis)}, 期望:{params.prt_num}, 跳过该帧")
                    continue

                # 目标检测
                amp_max_vr_unit = int(params.track_no_info[3])

                # 修正多普勒索引
                if amp_max_vr_unit > half_prt:
                    amp_max_vr_unit = amp_max_vr_unit - half_prt
                else:
                    amp_max_vr_unit = amp_max_vr_unit + half_prt

                # 转换为Python的0-based索引
                amp_max_vr_unit = amp_max_vr_unit - 1

                # 确保索引在有效范围内
                amp_max_vr_unit = np.clip(amp_max_vr_unit, 0, params.prt_num - 1)

                # 目标中心位于第16个距离单元
                center_local_bin = 15
                local_radius = 5

                # 计算局部检测窗口
                range_start_local = max(0, center_local_bin - local_radius)
                range_end_local = min(mtd_result.shape[0], center_local_bin + local_radius + 1)
                doppler_start = max(0, amp_max_vr_unit - local_radius)
                doppler_end = min(mtd_result.shape[1], amp_max_vr_unit + local_radius + 1)

                target_sig = mtd_result[range_start_local:range_end_local, doppler_start:doppler_end]

                # 检测峰值
                abs_target = np.abs(target_sig)
                if abs_target.size == 0:
                    continue

                max_idx = np.unravel_index(np.argmax(abs_target), abs_target.shape)
                amp_max_index_row, amp_max_index_col = max_idx

                # 获取目标全局距离单元索引
                global_range_bin = int(params.track_no_info[2])

                # 计算实际距离范围
                range_start_bin = global_range_bin - 15
                range_end_bin = global_range_bin + 15

                # 计算真实距离轴
                range_plot = np.arange(range_start_bin, range_end_bin + 1) * DELTA_R

                # 转换到全局距离位置
                detected_range_bin = range_start_local + amp_max_index_row
                if detected_range_bin >= len(range_plot):
                    continue

                # 安全地计算多普勒速度
                doppler_idx = doppler_start + amp_max_index_col
                if doppler_idx >= len(v_axis):
                    continue

                # 保存MTD处理结果
                rd_matrix = mtd_result
                velocity_axis = v_axis.reshape(-1)
                rd_matrix = np.abs(rd_matrix)

                velocity_index = np.where(velocity_axis == 0)[0][0]
                index = min(params.track_no_info[1], num_points)
                point_df = pl.read_csv(batch.point_file, has_header=True, separator=",", encoding="gbk")
                doppler_velocity = point_df["多普勒速度"][int(index) - 1]
                if abs(doppler_velocity) > 5:
                    rd_matrix[:, velocity_index-1:velocity_index+2] = 0

                # 2D-CFAR 检测和掩码生成
                processed_rd, target_row, target_col = cfar_detector_2d(
                    rd_matrix=rd_matrix,
                    detection_rows=np.arange(range_start_local, range_end_local),
                    detection_cols=np.arange(doppler_start, doppler_end),
                    num_guard_cells_range=3,
                    num_training_cells_range=5,
                    num_guard_cells_doppler=2,
                    num_training_cells_doppler=4,
                    cfar_threshold_factor=5
                )
                total += 1
                if processed_rd is None:
                    continue  # 未找到目标，跳过此帧
                detected += 1

                num_none_zero.append(np.count_nonzero(processed_rd))
                num_none_zero_row.append(np.count_nonzero(processed_rd[target_row-1:target_row+2, :]) / 3)
                missing_rate.append(detected / total)

                # rd_matrix[rd_matrix < np.percentile(rd_matrix, 5)] = 0
                velocity_mask = np.abs(velocity_axis) < 56
                velocity_axis = velocity_axis[velocity_mask]
                rd_matrix = rd_matrix[:, velocity_mask]
                rd_matrix = np.clip(rd_matrix, 1, 1e10)
                rd_matrix = 20 * np.log10(rd_matrix)

                rd_matrix = rd_matrix[:, :, None]
                if self.image_transform:
                    rd_matrix = self.image_transform(rd_matrix)
                rd_matrices.append(rd_matrix)
                point_index.append(index)

        if rd_matrices:
            rd_matrices = np.stack(rd_matrices, axis=0)
            point_index = np.array(point_index, dtype=np.int32)
        num_none_zero = np.array(num_none_zero, dtype=np.int32)
        num_none_zero_row = np.array(num_none_zero_row, dtype=np.float32)
        extra_features = np.stack([num_none_zero, num_none_zero_row], axis=1)
        missing_rate = np.array(missing_rate, dtype=np.float32)
        return rd_matrices, point_index, extra_features, missing_rate

    @staticmethod
    def collate_fn(batch):
        batch_files, point_indices, stacked_images, stacked_extra_features, stacked_missing_rate, image_masks, labels = \
            [], [], [], [], [], [], []
        for (batch_file, point_index, images, extra_features, missing_rate, image_mask, cls) in batch:
            if images is None:
                continue
            batch_files.append(batch_file)
            point_indices.append(point_index)
            stacked_images.append(images)
            stacked_extra_features.append(extra_features)
            stacked_missing_rate.append(missing_rate)
            image_masks.append(image_mask)
            labels.append(cls)
        point_indices = torch.from_numpy(np.stack(point_indices, axis=0))
        stacked_images = torch.from_numpy(np.stack(stacked_images, axis=0))
        stacked_extra_features = torch.from_numpy(np.stack(stacked_extra_features, axis=0))
        stacked_missing_rate = torch.from_numpy(np.stack(stacked_missing_rate, axis=0))
        image_masks = torch.from_numpy(np.stack(image_masks, axis=0))
        labels = torch.tensor(labels, dtype=torch.long)
        return batch_files, point_indices, stacked_images, stacked_extra_features, stacked_missing_rate, image_masks, labels


if __name__ == '__main__':
    from tqdm import tqdm
    import glob
    import re
    import plotly.graph_objects as go

    data_root = "D:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集"
    point_files = glob.glob(os.path.join(data_root, "点迹/PointTracks_*.txt"))
    missing = {i: [] for i in range(4)}
    for point_file in tqdm(point_files):
        match_result = re.match(r"PointTracks_(\d+)_(\d+)_(\d+).txt", os.path.basename(point_file))
        batch_id = int(match_result.group(1))
        label = int(match_result.group(2))
        if label > 4:
            continue
        num_points = int(match_result.group(3))
        raw_file = os.path.join(data_root, f"原始回波/{batch_id}_Label_{label}.dat")
        track_file = os.path.join(data_root, f"航迹/Tracks_{batch_id}_{label}_{num_points}.txt")
        batch_file = BatchFile(batch_id, label, raw_file, point_file, track_file)
        rd_matrices, ranges, velocities, missing_rates = process_batch(batch_file)
        if rd_matrices is None or len(rd_matrices) == 0:
            missing[label-1].append(1)
        else:
            missing[label-1].append(missing_rates[-1])
    for k, v in missing.items():
        fig = go.Figure(data=[go.Histogram(x=v, histnorm='probability', nbinsx=30)])
        fig.update_layout(
            title_text="缺失率分布",
            xaxis_title_text="缺失率",
            yaxis_title_text="占比",
            bargap=0.2,
            bargroupgap=0.1
        )
        fig.show()