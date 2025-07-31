import random

import torch
from torch.utils.data import Dataset

from .preprocess import *


TOTAL_FEATURES_PER_TIMESTEP = 28
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


class FusedDataset(Dataset):
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
        track_file = batch_file.track_file
        num_points = int(os.path.basename(point_file).split('_')[-1].split('.')[0])
        cls = batch_file.label - 1

        # load rd map
        images, point_index, extra_features = self._load_rd_map(batch_file, num_points)
        if len(images) == 0:
            return batch_file, None, None, None, None, None, None, cls

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
        elif images.shape[0] > self.image_seq_len:
            quantiles = np.linspace(0, 1, self.image_seq_len)
            indices = np.floor(quantiles * (images.shape[0] - 1)).astype(int)
            images = images[indices]
            point_index = point_index[indices]
            extra_features = extra_features[indices]
        assert images.shape[0] == self.image_seq_len, f"RD 图数量与预期不符: {images.shape[0]}, {self.image_seq_len}"

        # load point and track data
        merged_data = self._load_track_data(point_file, track_file)
        if len(merged_data) == 0:
            print('error')
            return batch_file, None, None, None, None, None, None, cls
        if merged_data.dtype != np.float32:
            merged_data = merged_data.astype(np.float32)
        track_mask = np.ones((self.track_seq_len,), dtype=np.int32)
        if merged_data.shape[0] < self.track_seq_len:
            track_mask[merged_data.shape[0]:] = 0
            merged_data = np.concatenate([
                merged_data,
                np.zeros((self.track_seq_len - merged_data.shape[0], TOTAL_FEATURES_PER_TIMESTEP))
            ], axis=0)
        elif merged_data.shape[0] > self.track_seq_len:
            merged_data = merged_data[:self.track_seq_len]
        assert merged_data.shape[0] == self.track_seq_len, f"点迹数量与预期不符: {merged_data.shape[0]}, {self.track_seq_len}"
        if self.track_transform:
            merged_data = self.track_transform(merged_data)

        return batch_file, point_index, images, merged_data, extra_features, image_mask, track_mask, cls

    def _load_rd_map(self, batch: BatchFile, num_points: int):
        """处理单个批次的数据"""
        # 打开原始数据文件
        frame_count = 0
        rd_matrices = []
        point_index = []
        # 额外特征
        num_none_zero = []
        num_none_zero_row = []


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

                if processed_rd is None:
                    continue  # 未找到目标，跳过此帧

                num_none_zero.append(np.count_nonzero(processed_rd))
                num_none_zero_row.append(np.count_nonzero(processed_rd[target_row-1:target_row+2, :]) / 3)

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
        return rd_matrices, point_index, extra_features

    @staticmethod
    def _load_track_data(point_filepath, track_filepath):
        """
        使用 polars 高效加载、合并数据并进行特征工程
        :param point_filepath: 点迹文件路径
        :param track_filepath: 航迹文件路径
        :return: 合并和处理后的特征数据 (NumPy Array)
        """
        try:
            # 1. 加载数据，并明确指定有表头
            preprocessed_data = TrajectoryDataProcessor(point_filepath, track_filepath).get_processed_data()
            df_point = pl.from_pandas(preprocessed_data['point_data'])
            df_track = pl.from_pandas(preprocessed_data['track_data'])

            # 2. 数据类型转换和合并
            df = df_point.join(df_track, on=["时间", "批号"], how="left").sort("时间")

            # 3. 特征工程 (基于用户提供的新逻辑)
            # 3.1 计算衍生的时序特征
            df = df.with_columns(
                (pl.col("滤波距离") * pl.col("滤波俯仰").radians().sin()).alias("高度"),
                (pl.col("X向速度").pow(2) + pl.col("Y向速度").pow(2)).sqrt().alias("水平速度"),
            )

            df = df.with_columns(
                pl.arctan2(pl.col("Z向速度"), pl.col("水平速度")).alias("爬升/俯冲角度_弧度"),
                (pl.col("和幅度") * pl.col("滤波距离").pow(4)).log10().alias("RCS"),
            )

            # 3.2 计算运动方向特征（用于鸟类识别）
            # 计算相邻时刻的方向变化角度
            df = df.with_columns(
                # 当前时刻的速度向量模长
                (pl.col("X向速度").pow(2) + pl.col("Y向速度").pow(2) + pl.col("Z向速度").pow(2)).sqrt().alias(
                    "三维速度模长"),
                # 计算时间差（用于角速度计算）
                (pl.col("时间").diff().dt.total_seconds()).alias("时间差"),
            )

            # 计算相邻速度向量的点积和夹角
            df = df.with_columns(
                # 相邻时刻速度向量点积
                (pl.col("X向速度") * pl.col("X向速度").shift(1) +
                 pl.col("Y向速度") * pl.col("Y向速度").shift(1) +
                 pl.col("Z向速度") * pl.col("Z向速度").shift(1)).alias("速度向量点积"),
                # 相邻时刻速度向量模长乘积
                (pl.col("三维速度模长") * pl.col("三维速度模长").shift(1)).alias("速度模长乘积"),
            )

            # 计算转向角度和角速度
            df = df.with_columns(
                # 转向角度 = arccos(dot_product / (|v1| * |v2|))
                pl.when(pl.col("速度模长乘积") > 1e-6)
                .then((pl.col("速度向量点积") / pl.col("速度模长乘积")).clip(-1.0, 1.0).arccos())
                .otherwise(0.0).alias("转向角度"),
            )

            df = df.with_columns(
                # 角速度 = 转向角度 / 时间差
                pl.when(pl.col("时间差") > 1e-6)
                .then(pl.col("转向角度") / pl.col("时间差"))
                .otherwise(0.0).alias("角速度"),
            )

            # 3.3 计算抖动特征（二阶差分）
            df = df.with_columns(
                # 多普勒速度的二阶差分
                pl.col("多普勒速度").diff().diff().alias("多普勒二阶差分"),
                # 幅度的二阶差分
                pl.col("和幅度").diff().diff().alias("幅度二阶差分"),
                # 位置的二阶差分
                pl.col("滤波距离").diff().diff().alias("距离二阶差分"),
                pl.col("滤波方位").diff().diff().alias("方位二阶差分"),
                pl.col("滤波俯仰").diff().diff().alias("俯仰二阶差分"),
            )

            # 3.4 mask机制的计算累积统计特征
            df = df.with_columns(
                # 原有特征
                pl.col("全速度").cum_min().alias("最小全速度"),
                (pl.col("全速度").cum_sum() / pl.col("全速度").cum_count()).alias("平均全速度"),
                (pl.col("水平速度").cum_sum() / pl.col("水平速度").cum_count()).alias("平均水平速度"),
                (pl.col("高度").cum_sum() / pl.col("高度").cum_count()).alias("平均高度"),
                pl.col("高度").cum_max().alias("最大高度"),
                pl.col("高度").cum_min().alias("最小高度"),
                pl.col("全速度").cum_max().alias("最大全速度"),
                pl.col("水平速度").cum_max().alias("最大水平速度"),
                pl.col("水平速度").cum_min().alias("最小水平速度"),

                # 新增：转向角度和角速度的累积统计
                (pl.col("转向角度").cum_sum() / pl.col("转向角度").cum_count()).alias("转向角度累积均值"),
                (pl.col("角速度").cum_sum() / pl.col("角速度").cum_count()).alias("角速度累积均值"),
            )

            # 计算累积波动范围和标准差
            df = df.with_columns(
                (pl.col("高度").cum_max() - pl.col("高度").cum_min()).alias("高度波动范围"),
                (pl.col("水平速度").cum_max() - pl.col("水平速度").cum_min()).alias("水平速度波动范围"),
                # 累积标准差（使用正确的方差公式）
                (pl.col("高度").pow(2).cum_sum() / pl.col("高度").cum_count() -
                 (pl.col("高度").cum_sum() / pl.col("高度").cum_count()).pow(2)).sqrt().alias("高度标准差"),

                # 转向角度和角速度的累积标准差
                (pl.col("转向角度").pow(2).cum_sum() / pl.col("转向角度").cum_count() -
                 (pl.col("转向角度").cum_sum() / pl.col("转向角度").cum_count()).pow(2)).sqrt().alias(
                    "转向角度累积标准差"),
                (pl.col("角速度").pow(2).cum_sum() / pl.col("角速度").cum_count() -
                 (pl.col("角速度").cum_sum() / pl.col("角速度").cum_count()).pow(2)).sqrt().alias("角速度累积标准差"),

                # 抖动指数（二阶差分的RMS）
                (pl.col("多普勒二阶差分").pow(2).cum_sum() / pl.col("多普勒二阶差分").cum_count()).sqrt().alias(
                    "多普勒抖动指数"),
                (pl.col("幅度二阶差分").pow(2).cum_sum() / pl.col("幅度二阶差分").cum_count()).sqrt().alias(
                    "幅度抖动指数"),
                ((pl.col("距离二阶差分").pow(2) + pl.col("方位二阶差分").pow(2) + pl.col("俯仰二阶差分").pow(2))
                 .cum_sum() / pl.col("距离二阶差分").cum_count()).sqrt().alias("位置抖动指数"),
            )

            # 4. 最终的特征
            final_feature_columns = [
                # 衍生特征
                "高度", "水平速度", "爬升/俯冲角度_弧度", "RCS",
                # 统计特征
                "最小全速度", "平均全速度", "平均水平速度", "平均高度",
                "最大高度", "最小高度", "高度波动范围", "高度标准差",
                "最大全速度", "最大水平速度", "最小水平速度", "水平速度波动范围",
                # 新增鸟类识别特征
                "转向角度累积均值", "转向角度累积标准差",
                "角速度累积均值", "角速度累积标准差",
                "多普勒抖动指数", "幅度抖动指数", "位置抖动指数",
                # 原始特征
                "俯仰", "多普勒速度", "和幅度", "信噪比", "原始点数量"
            ]
            df_final_features = df.select(final_feature_columns)

            # 5. 一次性填充所有因计算差分等产生的空值
            df_final_features = df_final_features.fill_null(0.0).fill_nan(0.0)

            return df_final_features.to_numpy(order='c').astype(np.float32)

        except Exception as e:
            print(f"处理文件时出错 {point_filepath}: {e}")
            return None


def collate_fn(batch):
    batch_files, point_indices, stacked_images, stacked_tracks, stacked_extra_features, image_masks, track_masks, labels = \
        [], [], [], [], [], [], [], []
    for (batch_file, point_index, images, merged_data, extra_features, image_mask, track_mask, cls) in batch:
        if images is None or merged_data is None:
            continue
        batch_files.append(batch_file)
        point_indices.append(point_index)
        stacked_images.append(images)
        stacked_tracks.append(merged_data)
        stacked_extra_features.append(extra_features)
        image_masks.append(image_mask)
        track_masks.append(track_mask)
        labels.append(cls)
    point_indices = torch.from_numpy(np.stack(point_indices, axis=0))
    stacked_images = torch.from_numpy(np.stack(stacked_images, axis=0))
    stacked_tracks = torch.from_numpy(np.stack(stacked_tracks, axis=0))
    stacked_extra_features = torch.from_numpy(np.stack(stacked_extra_features, axis=0))
    image_masks = torch.from_numpy(np.stack(image_masks, axis=0))
    track_masks = torch.from_numpy(np.stack(track_masks, axis=0))
    labels = torch.tensor(labels, dtype=torch.long)
    return batch_files, point_indices, stacked_images, stacked_tracks, stacked_extra_features, image_masks, track_masks, labels


if __name__ == '__main__':
    from fusion.utils.visualize import visualize_rd_matrix

    data_root = "D:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集"
    batch = 33
    label = 1
    batch_file = BatchFile(batch, label, os.path.join(data_root, f"原始回波/{batch}_Label_{label}.dat"),
                           os.path.join(data_root, f"点迹/PointTracks_{batch}_{label}_23.txt"),
                           os.path.join(data_root, f"航迹/Tracks_{batch}_{label}_23.txt"))
    rd_matrices, ranges, velocities = process_batch(batch_file)
    for i in range(min(50, len(rd_matrices))):
        visualize_rd_matrix(rd_matrices[i], ranges[i], velocities[i], batch, label, i)