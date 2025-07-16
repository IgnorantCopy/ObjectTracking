import os
import re
import struct
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl
from scipy import signal
from scipy.fft import fft, fftshift
from dataclasses import dataclass
from pathlib import Path


POINT_TRACK_HEADER = ["点时间", "批号", "距离", "方位", "俯仰", "多普勒速度", "和幅度", "信噪比", "原始点数量"]
TRACK_HEADER = ["点时间", "批号", "滤波距离", "滤波方位", "滤波俯仰", "全速度", "X向速度", "Y向速度", "Z向速度", "航向"]

NUMERICAL_POINT_FEATURES = ["距离", "方位", "俯仰", "多普勒速度", "和幅度", "信噪比", "原始点数量"]  # 7
NUMERICAL_TRACK_FEATURES = ["滤波距离", "滤波方位", "滤波俯仰", "全速度", "X向速度", "Y向速度", "Z向速度", "航向"]  # 8
TOTAL_FEATURES_PER_TIMESTEP = len(NUMERICAL_POINT_FEATURES) + len(NUMERICAL_TRACK_FEATURES)  # 15

ABNORMAL_BATCH_ID = [1451, 1452, 1457, 1462, 1467, 1469, 1473, 1478, 1484, 1487, 1488, 1490, 1494, 1496, 1497, 1500]

FS = 20e6  # 采样率 (20 MHz)
C = 3e8    # 光速 (m/s)
DELTA_R = C / (2 * FS)  # 距离分辨率


class ColPoint:
    """点迹数据列索引"""
    Time = 0       # 点时间
    TrackID = 1    # 航迹批号
    R = 2          # 距离
    AZ = 3         # 方位
    EL = 4         # 俯仰
    Doppler = 5    # 多普勒速度
    Amp = 6        # 和幅度
    SNR = 7        # 信噪比
    PointNum = 8   # 原始点数量


class ColTrack:
    """航迹数据列索引"""
    Time = 0       # 点时间
    TrackID = 1    # 航迹批号
    R = 2          # 滤波距离
    AZ = 3         # 滤波方位
    EL = 4         # 滤波俯仰
    Speed = 5      # 全速度
    Vx = 6         # X向速度(东)
    Vy = 7         # Y向速度(北)
    Vz = 8         # Z向速度(天)
    Head = 9       # 航向角


@dataclass
class BatchFile:
    """批次文件信息"""
    batch_num: int           # 航迹批号
    label: int              # 目标类型标签
    raw_file: str          # 原始回波文件路径
    point_file: str        # 点迹文件路径
    track_file: str        # 航迹文件路径


@dataclass
class Parameters:
    """雷达参数"""
    e_scan_az: float       # 方位角
    track_no_info: np.ndarray  # 航迹信息
    freq: float           # 频率
    cpi_count: int       # CPI流水号
    prt_num: int         # PRT数目
    prt: float          # PRT值
    data_length: int    # 距离维采样点数


def read_raw_data(fid):
    """
    读取解析原始回波数据
    :param fid: 文件句柄
    :return: 参数对象和数据数组
    """
    FRAME_HEAD = 0xFA55FA55
    FRAME_END = 0x55FA55FA

    # 读取帧头
    try:
        head_bytes = fid.read(4)
        if len(head_bytes) < 4:
            return None, None
        head_find = struct.unpack('<I', head_bytes)[0]  # 使用小端序
    except struct.error:
        return None, None

    # 查找帧头 - 更接近MATLAB逻辑
    file_size = os.fstat(fid.fileno()).st_size
    while head_find != FRAME_HEAD and fid.tell() < file_size:
        fid.seek(-3, 1)  # 回退3个字节
        try:
            head_bytes = fid.read(4)
            if len(head_bytes) < 4:
                return None, None
            head_find = struct.unpack('<I', head_bytes)[0]
        except struct.error:
            return None, None

    if head_find != FRAME_HEAD:
        return None, None

    # 读取帧长度
    try:
        length_bytes = fid.read(4)
        if len(length_bytes) < 4:
            return None, None
        frame_data_length = struct.unpack('<I', length_bytes)[0] * 4  # 使用小端序
    except struct.error:
        return None, None

    if frame_data_length <= 0 or frame_data_length > 1000000:
        return None, None

    # 检查帧尾 - 更接近MATLAB逻辑
    current_pos = fid.tell()

    try:
        fid.seek(current_pos + frame_data_length - 12, 0)  # 偏移到结尾
        end_bytes = fid.read(4)
        if len(end_bytes) < 4:
            return None, None
        end_find = struct.unpack('<I', end_bytes)[0]
    except (struct.error, OSError):
        return None, None

    # 验证帧头和帧尾
    while head_find != FRAME_HEAD or end_find != FRAME_END:
        fid.seek(-frame_data_length + 1, 1)  # 指针偏移

        try:
            head_bytes = fid.read(4)
            if len(head_bytes) < 4:
                return None, None
            head_find = struct.unpack('<I', head_bytes)[0]

            length_bytes = fid.read(4)
            if len(length_bytes) < 4:
                return None, None
            frame_data_length = struct.unpack('<I', length_bytes)[0] * 4

            if frame_data_length <= 0 or frame_data_length > 1000000:
                return None, None

            fid.seek(frame_data_length - 8, 1)
            end_bytes = fid.read(4)
            if len(end_bytes) < 4:
                return None, None
            end_find = struct.unpack('<I', end_bytes)[0]

        except struct.error:
            return None, None

        if fid.tell() >= file_size and (head_find != FRAME_HEAD or end_find != FRAME_END):
            print('未找到满足报文格式的数据')
            return None, None

    # 回到数据开始位置
    fid.seek(-frame_data_length + 4, 1)

    # 读取参数
    try:
        # 读取前3个uint32
        data_temp1_bytes = fid.read(12)
        if len(data_temp1_bytes) < 12:
            return None, None
        data_temp1 = np.frombuffer(data_temp1_bytes, dtype='<u4')  # 小端序uint32

        e_scan_az = data_temp1[1] * 0.01
        point_num_in_bowei = data_temp1[2]

        # 添加point_num合理性检查
        if point_num_in_bowei < 0 or point_num_in_bowei > 1000:
            print(f"点数异常: {point_num_in_bowei}")
            return None, None

        # 读取航迹信息和其他参数
        param_count = point_num_in_bowei * 4 + 5
        param_bytes = fid.read(param_count * 4)
        if len(param_bytes) < param_count * 4:
            return None, None
        data_temp = np.frombuffer(param_bytes, dtype='<u4')

        # 提取航迹信息
        if point_num_in_bowei > 0:
            track_no_info = data_temp[:point_num_in_bowei * 4]
        else:
            track_no_info = np.array([], dtype=np.uint32)

        # 提取其他参数
        base_idx = point_num_in_bowei * 4
        params = Parameters(
            e_scan_az=e_scan_az,
            track_no_info=track_no_info,
            freq=data_temp[base_idx] * 1e6,
            cpi_count=data_temp[base_idx + 1],
            prt_num=data_temp[base_idx + 2],
            prt=data_temp[base_idx + 3] * 0.0125e-6,
            data_length=data_temp[base_idx + 4]
        )

        # 参数验证
        if params.prt_num <= 0 or params.prt_num > 10000:
            print(f"PRT_num异常: {params.prt_num}")
            return None, None
        if params.prt <= 0 or params.prt > 1:
            print(f"PRT异常: {params.prt}")
            return None, None
        if params.freq <= 0 or params.freq > 1e12:
            print(f"频率异常: {params.freq}")
            return None, None

        # 读取IQ数据
        iq_data_len = params.prt_num * 31 * 2
        data_bytes = fid.read(iq_data_len * 4)
        if len(data_bytes) < iq_data_len * 4:
            print(f"IQ数据长度不足: 期望{iq_data_len * 4}, 实际{len(data_bytes)}")
            return None, None

        data_out_temp = np.frombuffer(data_bytes, dtype='<f4')  # 小端序float32

        # 重构复数数据
        data_out_real = data_out_temp[::2]
        data_out_imag = data_out_temp[1::2]
        data_out_complex = data_out_real + 1j * data_out_imag
        data_out = data_out_complex.reshape(31, params.prt_num, order='F')

        # 跳过帧尾
        fid.seek(4, 1)

        return params, data_out

    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        return None, None


def process_batch(batch: BatchFile):
    """处理单个批次的数据"""
    # 打开原始数据文件
    frame_count = 0
    rd_matrices = []
    ranges = []
    velocities = []
    try:
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

                try:
                    # MTD处理
                    # ===== 关键修改 1：修正加窗操作 =====
                    distance_bins = data.shape[0]  # 距离单元数 (31)
                    prt_bins = data.shape[1]  # PRT数
                    # 生成泰勒窗 - 在距离维度加窗（窗长 = 距离单元数）
                    mtd_win = signal.windows.taylor(distance_bins, nbar=4, sll=30, norm=False)
                    # 将窗函数转换为列向量 (31×1)
                    mtd_win_col = mtd_win.reshape(-1, 1)
                    # 在PRT维度（列方向）重复窗函数 (31×N)
                    coef_mtd_2d = np.repeat(mtd_win_col, prt_bins, axis=1)
                    # 加窗处理
                    data_windowed = data * coef_mtd_2d
                    # FFT处理 - 在慢时间维度（轴1）进行FFT
                    mtd_result = fftshift(fft(data_windowed, axis=1), axes=1)

                    # 计算多普勒速度轴 - 修复溢出问题
                    try:
                        delta_v = C / (2 * params.prt_num * params.prt * params.freq)

                        # 检查delta_v是否有效
                        if not np.isfinite(delta_v) or delta_v <= 0 or delta_v > 10000:
                            print(f"警告：帧 {frame_count} delta_v异常: {delta_v}, 跳过该帧")
                            continue

                        # 修复溢出问题 - 使用更安全的方式
                        half_prt = prt_bins // 2

                        # 检查half_prt是否合理
                        if half_prt <= 0 or half_prt > 10000:
                            print(f"警告：帧 {frame_count} half_prt异常: {half_prt}, 跳过该帧")
                            continue

                        # 使用int32避免溢出
                        v_axis = np.linspace(-prt_bins / 2 * delta_v,
                                             prt_bins / 2 * delta_v,
                                             prt_bins,
                                             endpoint=False)

                        # 检查v_axis是否有效
                        if not np.all(np.isfinite(v_axis)) or len(v_axis) != params.prt_num:
                            print(
                                f"警告：帧 {frame_count} v_axis异常，长度:{len(v_axis)}, 期望:{params.prt_num}, 跳过该帧")
                            continue

                    except Exception as e:
                        print(f"警告：帧 {frame_count} 计算速度轴时出错: {str(e)}")
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
                    range_axis = range_plot
                    velocity_axis = v_axis
                    velocity_mask = np.abs(velocity_axis) < 56
                    velocity_axis = velocity_axis[velocity_mask]
                    rd_matrix = rd_matrix[:, np.reshape(velocity_mask, -1)]
                    rd_matrix = np.abs(rd_matrix)
                    rd_matrix = np.clip(rd_matrix, 1e-10, 1000)
                    rd_matrix = 20 * np.log10(rd_matrix)
                    velocity_index = np.where(np.reshape(velocity_axis, -1) == 0)[0][0]
                    rd_matrix[:, velocity_index - 4:velocity_index + 3] = 0
                    rd_matrix[rd_matrix < np.percentile(rd_matrix, 5)] = 0
                    rd_matrices.append(rd_matrix)
                    ranges.append(range_axis)
                    velocities.append(velocity_axis)

                except Exception as e:
                    # 静默跳过有问题的帧，避免过多错误输出
                    continue

    except Exception as e:
        raise ValueError(f"读取原始数据文件失败：{str(e)}")

    return rd_matrices, ranges, velocities


def get_batch_file_list(root_dir: str):
    """
    获取批量处理文件列表
    :param root_dir: 数据根目录
    :return: 批次文件列表
    """
    iq_dir = os.path.join(root_dir, "原始回波")
    track_dir = os.path.join(root_dir, "航迹")
    point_dir = os.path.join(root_dir, "点迹")

    if not all(os.path.isdir(d) for d in [iq_dir, track_dir, point_dir]):
        raise ValueError("错误！数据根目录下需包含原始回波、点迹、航迹三个子文件夹。")

    batch_files = []
    # 遍历原始回波文件
    for raw_file in os.listdir(iq_dir):
        if not raw_file.endswith('.dat'):
            continue

        # 解析文件名
        match = re.match(r'^(\d+)_Label_(\d+)\.dat$', raw_file)
        if not match:
            continue

        batch_num = int(match.group(1))
        label = int(match.group(2))

        # 查找对应的点迹和航迹文件
        point_pattern = f'PointTracks_{batch_num}_{label}_*.txt'
        track_pattern = f'Tracks_{batch_num}_{label}_*.txt'

        point_files = list(Path(point_dir).glob(point_pattern))
        track_files = list(Path(track_dir).glob(track_pattern))

        if point_files and track_files:
            batch_files.append(BatchFile(
                batch_num=batch_num,
                label=label,
                raw_file=os.path.join(iq_dir, raw_file),
                point_file=str(point_files[0]),
                track_file=str(track_files[0])
            ))
        else:
            missing_point = len(point_files) == 0
            missing_track = len(track_files) == 0
            msg = f"警告：批号 {batch_num}、标签 {label} 的"
            if missing_point and missing_track:
                msg += "点迹和航迹文件均未找到，已跳过。"
            elif missing_point:
                msg += "点迹文件未找到，已跳过。"
            else:
                msg += "航迹文件未找到，已跳过。"
            print(msg)

    if not batch_files:
        raise ValueError("未找到符合命名规则的批量处理文件（需为：航迹批号_Label_目标类型标签.dat）！")

    return batch_files


def split_train_val(data_root: str, num_classes, val_ratio=0.2, shuffle=True):
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
        images, point_index = self._process_batch(batch_file, num_points)
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
        elif images.shape[0] > self.image_seq_len:
            images = images[:self.image_seq_len]
            point_index = point_index[:self.image_seq_len]
        assert images.shape[0] == self.image_seq_len, f"RD 图数量与预期不符: {images.shape[0]}, {self.image_seq_len}"

        # load point and track data
        merged_data = self._load_and_merge_data(point_file, track_file)
        assert merged_data is not None and merged_data.shape[0] > 0, f"读取或合并文件失败: {point_file}, {track_file}"
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

        return batch_file, point_index, images, merged_data, image_mask, track_mask, cls

    def _process_batch(self, batch: BatchFile, num_points: int):
        """处理单个批次的数据"""
        # 打开原始数据文件
        frame_count = 0
        rd_matrices = []
        point_index = []
        try:
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

                    try:
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
                        try:
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

                        except Exception as e:
                            print(f"警告：帧 {frame_count} 计算速度轴时出错: {str(e)}")
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
                        velocity_axis = v_axis
                        velocity_mask = np.reshape(np.abs(velocity_axis) < 56, -1)
                        rd_matrix = rd_matrix[:, velocity_mask]
                        rd_matrix = np.abs(rd_matrix)
                        rd_matrix = np.clip(rd_matrix, 1e-10, 1000)
                        rd_matrix = 20 * np.log10(rd_matrix)
                        velocity_index = np.where(np.reshape(velocity_axis, -1) == 0)[0][0]
                        rd_matrix[:, velocity_index - 4:velocity_index + 3] = 0
                        rd_matrix[rd_matrix < np.percentile(rd_matrix, 5)] = 0
                        rd_matrix = rd_matrix[:, :, None]
                        if self.image_transform:
                            rd_matrix = self.image_transform(rd_matrix)
                        index = min(params.track_no_info[1], num_points)
                        rd_matrices.append(rd_matrix)
                        point_index.append(index)

                    except Exception as e:
                        # 静默跳过有问题的帧，避免过多错误输出
                        continue

        except Exception as e:
            raise ValueError(f"读取原始数据文件失败：{str(e)}")

        rd_matrices = np.stack(rd_matrices, axis=0)
        point_index = np.array(point_index, dtype=np.int32)
        return rd_matrices, point_index

    @staticmethod
    def _load_and_merge_data(point_track_filepath, track_filepath):
        """
        加载点迹和航迹数据，并根据时间戳和批号进行合并。
        由于时间戳和批号一一对应，直接水平拼接数值特征。
        """
        try:
            # 使用Polars读取CSV，指定编码和分隔符
            # has_header=True 因为文件现在有header了
            df_point = pl.read_csv(point_track_filepath, separator=",", has_header=True, encoding="gbk")
            df_track = pl.read_csv(track_filepath, separator=",", has_header=True, encoding="gbk")

            # --- 关键修改 1: 在读取后立即处理 NaN/Null 值 ---
            # 填充所有数值列的 null 为 0.0。对于非数值列，此操作无效。
            # 也可以考虑更复杂的填充策略，但对于特征数据，0.0通常是安全的初始选择。
            # 确保只对实际存在的数值列进行填充
            for col in NUMERICAL_POINT_FEATURES:
                if col in df_point.columns:
                    df_point = df_point.with_columns(pl.col(col).fill_null(0.0).cast(pl.Float64))  # 强制转换为浮点数
            for col in NUMERICAL_TRACK_FEATURES:
                if col in df_track.columns:
                    df_track = df_track.with_columns(pl.col(col).fill_null(0.0).cast(pl.Float64))  # 强制转换为浮点数

            # 检查时间戳和批号是否完全一致，这是合并的前提
            if "批号" not in df_point.columns or "批号" not in df_track.columns:
                print(
                    f"警告: 文件 {os.path.basename(point_track_filepath)} 或 {os.path.basename(track_filepath)} 缺少'批号'列。")
                return None

            # 转换为字符串或统一类型再比较，防止类型不一致导致的比较问题
            # 注意: 如果原始数据中'点时间'或'批号'列存在Null，这里也可能导致问题，
            # 但通常这些关键ID列不应该有Null。如果确实有，也需要fill_null。
            if not df_point["点时间"].equals(df_track["点时间"]) or not df_point["批号"].cast(pl.Utf8).equals(
                    df_track["批号"].cast(pl.Utf8)):
                print(
                    f"警告: {os.path.basename(point_track_filepath)} 与 {os.path.basename(track_filepath)} 时间戳或批号不完全一致，跳过此文件对。")
                return None

            # 水平拼接数值特征列
            # 确保所有特征列都存在
            missing_point_features = [f for f in NUMERICAL_POINT_FEATURES if f not in df_point.columns]
            missing_track_features = [f for f in NUMERICAL_TRACK_FEATURES if f not in df_track.columns]

            if missing_point_features or missing_track_features:
                print(
                    f"警告: 文件 {os.path.basename(point_track_filepath)} 或 {os.path.basename(track_filepath)} 缺少必要的特征列。")
                if missing_point_features:
                    print(f"  点迹缺少: {missing_point_features}")
                if missing_track_features:
                    print(f"  航迹缺少: {missing_track_features}")
                return None

            merged_df = pl.concat([df_point[NUMERICAL_POINT_FEATURES], df_track[NUMERICAL_TRACK_FEATURES]],
                                  how="horizontal")

            # --- 关键修改 2: 在转换为 NumPy 之前，再次检查并处理可能的 NaN/Inf ---
            # Polars的DataFrame在转换为NumPy时，如果内部有NaN/Inf，会直接传递。
            # 这一步是双重保险，以防Polars的fill_null没有处理到所有情况(例如，数据类型推断问题)。
            merged_np = merged_df.to_numpy()

            if np.isnan(merged_np).any() or np.isinf(merged_np).any():
                print(
                    f"DEBUG: 在文件 {os.path.basename(point_track_filepath)} 合并后的NumPy数组中检测到 NaN/Inf。正在填充...")
                merged_np[np.isnan(merged_np)] = 0.0
                merged_np[np.isinf(merged_np)] = 0.0  # 处理无穷大值

            return merged_np

        except Exception as e:
            print(f"读取或合并文件失败: {point_track_filepath}, {track_filepath}. 错误: {e}")
            return None


def collate_fn(batch):
    batch_files, point_indices, stacked_images, stacked_tracks, image_masks, track_masks, labels = [], [], [], [], [], [], []
    for (batch_file, point_index, images, merged_data, image_mask, track_mask, cls) in batch:
        batch_files.append(batch_file)
        point_indices.append(point_index)
        stacked_images.append(images)
        stacked_tracks.append(merged_data)
        image_masks.append(image_mask)
        track_masks.append(track_mask)
        labels.append(cls)
    point_indices = torch.from_numpy(np.stack(point_indices, axis=0))
    stacked_images = torch.from_numpy(np.stack(stacked_images, axis=0))
    stacked_tracks = torch.from_numpy(np.stack(stacked_tracks, axis=0))
    image_masks = torch.from_numpy(np.stack(image_masks, axis=0))
    track_masks = torch.from_numpy(np.stack(track_masks, axis=0))
    labels = torch.tensor(labels, dtype=torch.long)
    return batch_files, point_indices, stacked_images, stacked_tracks, image_masks, track_masks, labels


if __name__ == '__main__':
    from visualize import visualize_rd_matrix

    data_root = "D:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集"
    batch = 1333
    label = 2
    batch_file = BatchFile(batch, label, os.path.join(data_root, f"原始回波/{batch}_Label_{label}.dat"),
                           os.path.join(data_root, f"点迹/PointTracks_{batch}_{label}_25.txt"),
                           os.path.join(data_root, f"航迹/Tracks_{batch}_{label}_25.txt"))
    rd_matrices, ranges, velocities = process_batch(batch_file)
    visualize_rd_matrix(rd_matrices[9], ranges[9], velocities[9], batch, label, 9)