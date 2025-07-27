import os
import re
import glob
import struct
import numpy as np
import pandas as pd
import polars as pl
from scipy import signal, interpolate
from scipy.fft import fft, fftshift
from scipy.stats import zscore
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


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


def cfar_detector_2d(rd_matrix, cfar_threshold_factor,
                     detection_rows, detection_cols,
                     num_guard_cells_range, num_training_cells_range,
                     num_guard_cells_doppler, num_training_cells_doppler):
    """
    对RD矩阵执行2D-CFAR检测

    :param rd_matrix: 输入的 RD 矩阵 (linear)
    :param detection_rows: 距离维检测区域行索引范围
    :param detection_cols: 多普勒维检测区域列索引范围
    :param num_guard_cells_range: 距离维保护单元数 (单侧)
    :param num_training_cells_range: 距离维训练单元数 (单侧) - 这里会被忽略
    :param num_guard_cells_doppler: 多普勒维保护单元数 (单侧)
    :param num_training_cells_doppler: 多普勒维训练单元数 (单侧)
    :param cfar_threshold_factor: CFAR门限因子
    :return: 如果找到目标，返回一个应用了十字掩码的RD矩阵；否则返回None
    """

    if rd_matrix.size == 0:
        return None

    rows, cols = rd_matrix.shape

    zero_vel_idx = cols // 2

    # 排除零速度附近的杂波区域
    clutter_width = 3
    non_clutter_mask = np.ones(cols, dtype=bool)
    non_clutter_mask[max(0, zero_vel_idx - clutter_width):min(cols, zero_vel_idx + clutter_width + 1)] = False

    if not np.any(non_clutter_mask):
        print("  -> 返回None: 无有效检测区域")
        return None

    # 调用修正的CFAR检测
    target_detected, target_mask, target_row, target_col = func_ca_cfar_detect_all_targets_new(
        rd_matrix,
        detection_rows,
        detection_cols,
        num_guard_cells_range,
        num_guard_cells_doppler,
        num_training_cells_range,
        num_training_cells_doppler,
        cfar_threshold_factor
    )

    if target_detected:
        result = rd_matrix * target_mask.astype(float)
        return result, target_row, target_col
    else:
        return None, None, None


def func_ca_cfar_detect_all_targets_new(data_power, detection_rows, detection_cols,
                                        Gr, Gd, Tr, Td, threshold_factor):
    """
    严格按照MATLAB版本实现的CA-CFAR检测
    """
    num_rows, num_cols = data_power.shape

    # 初始化
    local_detection_area = data_power[np.ix_(detection_rows, detection_cols)]
    local_num_rows, local_num_cols = local_detection_area.shape
    status_map = np.zeros((local_num_rows, local_num_cols), dtype=int)
    output_mask = np.zeros_like(data_power, dtype=bool)

    # 按幅度降序排序
    sorted_indices = np.argsort(local_detection_area.ravel())[::-1]
    rows_in_local, cols_in_local = np.unravel_index(sorted_indices, local_detection_area.shape)

    for k in range(len(sorted_indices)):
        local_row = rows_in_local[k]
        local_col = cols_in_local[k]

        if status_map[local_row, local_col] != 0:
            continue

        # 转换为全局坐标
        i = detection_rows[local_row]
        j = detection_cols[local_col]
        cut_power = data_power[i, j]

        is_target = True

        # 距离向上方参考单元
        sum_noise_upper, count_noise_upper = 0, 0
        for r_ref in range(i - Gr - Td * 2, i - Gr):
            if 0 <= r_ref < num_rows:
                # 检查是否在检测区域内且已被处理
                if detection_rows[0] <= r_ref <= detection_rows[-1]:
                    local_r = r_ref - detection_rows[0]
                    if status_map[local_r, local_col] in [1, 2]:
                        continue
                sum_noise_upper += data_power[r_ref, j]
                count_noise_upper += 1

        if count_noise_upper == 0 or cut_power <= threshold_factor * (sum_noise_upper / count_noise_upper):
            is_target = False

        if not is_target:
            status_map[local_row, local_col] = 2
            continue

        # 距离向下方参考单元
        sum_noise_lower, count_noise_lower = 0, 0
        for r_ref in range(i + Gr + 1, i + Gr + Td * 2 + 1):
            if 0 <= r_ref < num_rows:
                if detection_rows[0] <= r_ref <= detection_rows[-1]:
                    local_r = r_ref - detection_rows[0]
                    if status_map[local_r, local_col] in [1, 2]:
                        continue
                sum_noise_lower += data_power[r_ref, j]
                count_noise_lower += 1

        if count_noise_lower == 0 or cut_power <= threshold_factor * (sum_noise_lower / count_noise_lower):
            is_target = False

        if not is_target:
            status_map[local_row, local_col] = 2
            continue

        # 多普勒向参考单元（左右取较小值）
        sum_noise_left, count_noise_left = 0, 0
        for d_ref in range(j - Gd - Td, j - Gd):
            if 0 <= d_ref < num_cols:
                if detection_cols[0] <= d_ref <= detection_cols[-1]:
                    local_d = d_ref - detection_cols[0]
                    if status_map[local_row, local_d] in [1, 2]:
                        continue
                sum_noise_left += data_power[i, d_ref]
                count_noise_left += 1

        sum_noise_right, count_noise_right = 0, 0
        for d_ref in range(j + Gd + 1, j + Gd + Td + 1):
            if 0 <= d_ref < num_cols:
                if detection_cols[0] <= d_ref <= detection_cols[-1]:
                    local_d = d_ref - detection_cols[0]
                    if status_map[local_row, local_d] in [1, 2]:
                        continue
                sum_noise_right += data_power[i, d_ref]
                count_noise_right += 1

        # 取左右两侧噪声估计的较小值
        noise_avg_left = sum_noise_left / count_noise_left if count_noise_left > 0 else float('inf')
        noise_avg_right = sum_noise_right / count_noise_right if count_noise_right > 0 else float('inf')
        noise_avg_doppler = min(noise_avg_left, noise_avg_right)

        if np.isinf(noise_avg_doppler) or cut_power <= threshold_factor * noise_avg_doppler:
            is_target = False

        # 最终判定
        if is_target:
            status_map[local_row, local_col] = 1
            output_mask[i, j] = True
        else:
            status_map[local_row, local_col] = 2

    return np.any(output_mask), output_mask, i, j


def process_batch(batch: BatchFile):
    """处理单个批次的数据"""
    # 打开原始数据文件
    frame_count = 0
    rd_matrices = []
    ranges = []
    velocities = []

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
            v_axis = np.linspace(-prt_bins / 2 * delta_v,prt_bins / 2 * delta_v, prt_bins, endpoint=False)

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
            range_axis = range_plot
            velocity_axis = v_axis.reshape(-1)
            rd_matrix = np.abs(rd_matrix)

            velocity_index = np.where(velocity_axis == 0)[0][0]
            point_df = pl.read_csv(batch.point_file, has_header=True, separator=",", encoding="gbk")
            index = min(params.track_no_info[1], len(point_df))
            doppler_velocity = point_df["多普勒速度"][int(index) - 1]
            if abs(doppler_velocity) > 5:
                rd_matrix[:, velocity_index - 1:velocity_index + 2] = 0

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
                print(f"帧 {frame_count} 未找到目标，跳过该帧")
                continue  # 未找到目标，跳过此帧

            rd_matrix = processed_rd

            # rd_matrix[rd_matrix < np.percentile(rd_matrix, 5)] = 0
            velocity_mask = np.abs(velocity_axis) < 56
            velocity_axis = velocity_axis[velocity_mask]
            rd_matrix = rd_matrix[:, velocity_mask]
            rd_matrix = np.clip(rd_matrix, 1, 1e10)
            rd_matrix = 20 * np.log10(rd_matrix)

            rd_matrices.append(rd_matrix)
            ranges.append(range_axis)
            velocities.append(velocity_axis)



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


class TrajectoryDataProcessor(object):
    """
    点迹和航迹数据处理类，包含异常值检测和插值修复
    """

    def __init__(self, point_file_path, track_file_path, verbose=False,
                 outlier_threshold=3.0, interpolation_method='linear',
                 velocity_threshold=100.0, doppler_threshold=50.0):
        """
        初始化数据处理器

        Args:
            point_file_path: 点迹文件路径
            track_file_path: 航迹文件路径
            outlier_threshold: Z-score异常值阈值
            interpolation_method: 插值方法 ('linear', 'cubic', 'quadratic')
            velocity_threshold: 速度异常值阈值
            doppler_threshold: 多普勒速度异常值阈值
        """
        self.point_file_path = point_file_path
        self.track_file_path = track_file_path
        self.outlier_threshold = outlier_threshold
        self.interpolation_method = interpolation_method
        self.velocity_threshold = velocity_threshold
        self.doppler_threshold = doppler_threshold
        self.verbose = verbose

        # 数据存储
        self.point_data = None
        self.track_data = None
        self.processed_point_data = None
        self.processed_track_data = None

        # 处理数据
        if point_file_path or track_file_path:
            self.load_and_process_data()

    def load_and_process_data(self):
        """加载并处理数据"""

        # 加载原始数据
        if self.point_file_path:
            self.point_data = self._load_point_data()

        if self.track_file_path:
            self.track_data = self._load_track_data()

        # 处理异常值
        if self.point_data is not None:
            self.processed_point_data = self._process_point_outliers()

        if self.track_data is not None:
            self.processed_track_data = self._process_track_outliers()

    def _load_point_data(self):
        """加载点迹数据"""
        columns = ['时间', '批号', '距离', '方位', '俯仰', '多普勒速度', '和幅度', '信噪比', '原始点数量']

        try:
            data = pd.read_csv(self.point_file_path, encoding='gbk', header=0, names=columns)
            # 转换时间格式
            data['时间'] = pd.to_datetime(data['时间'], format='%H:%M:%S.%f')
            return data
        except Exception as e:
            print(f"加载点迹数据失败: {e}")
            return None

    def _load_track_data(self):
        """加载航迹数据"""
        columns = ['时间', '批号', '滤波距离', '滤波方位', '滤波俯仰', '全速度',
                   'X向速度', 'Y向速度', 'Z向速度', '航向']

        try:
            data = pd.read_csv(self.track_file_path, encoding='gbk', header=0, names=columns)
            # 转换时间格式
            data['时间'] = pd.to_datetime(data['时间'], format='%H:%M:%S.%f')
            return data
        except Exception as e:
            print(f"加载航迹数据失败: {e}")
            return None

    def _detect_outliers_zscore(self, series, threshold=None):
        """使用Z-score检测异常值"""
        if threshold is None:
            threshold = self.outlier_threshold

        if len(series) <= 1:
            return pd.Series([False] * len(series), index=series.index)

        z_scores = np.abs(zscore(series, nan_policy='omit'))
        return pd.Series(z_scores > threshold, index=series.index)

    def _detect_outliers_iqr(self, series):
        """使用IQR方法检测异常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)

    def _detect_velocity_outliers(self, series, threshold):
        """检测速度异常值（绝对值过大）"""
        return np.abs(series) > threshold

    def _extrapolate_outliers(self, series, outlier_mask):
        """
        使用外插法修复异常值 - 只使用异常值之前的数据进行预测
        """
        if outlier_mask.sum() == 0:
            return series.copy()

        series_copy = series.copy()

        # 按时间顺序处理每个异常点
        indices = np.arange(len(series))
        for i in indices[outlier_mask]:
            # 只使用当前点之前的有效数据点
            prior_indices = indices[:i]
            prior_valid_mask = ~outlier_mask.iloc[prior_indices]
            prior_valid_indices = prior_indices[prior_valid_mask]

            # 如果之前没有足够的有效点，尝试使用全局有效点的均值
            if len(prior_valid_indices) < 2:
                all_valid_indices = indices[~outlier_mask]
                if len(all_valid_indices) >= 1:
                    # 使用所有有效点的均值
                    fill_value = series.iloc[all_valid_indices].mean()
                    if pd.isna(fill_value):
                        fill_value = 0
                    series_copy.iloc[i] = fill_value
                else:
                    # 没有有效点时使用0填充
                    series_copy.iloc[i] = 0
                continue

            # 获取之前的有效点的数据
            x_prior = prior_valid_indices
            y_prior = series.iloc[prior_valid_indices].values

            try:
                # 根据插值方法选择不同的模型
                if self.interpolation_method == 'linear':
                    # 线性外插
                    model = np.polyfit(x_prior, y_prior, 1)
                    predicted_value = np.polyval(model, i)
                elif self.interpolation_method == 'quadratic' and len(x_prior) >= 3:
                    # 二次多项式外插
                    model = np.polyfit(x_prior, y_prior, 2)
                    predicted_value = np.polyval(model, i)
                elif self.interpolation_method == 'cubic' and len(x_prior) >= 4:
                    # 三次多项式外插
                    model = np.polyfit(x_prior, y_prior, 3)
                    predicted_value = np.polyval(model, i)
                elif self.interpolation_method == 'exp':
                    # 指数模型外插 (适用于指数型增长数据)
                    # 对y取对数，再用线性拟合
                    if np.all(y_prior > 0):  # 确保所有值为正
                        log_y = np.log(y_prior)
                        model = np.polyfit(x_prior, log_y, 1)
                        predicted_value = np.exp(np.polyval(model, i))
                    else:
                        # 若有非正值，使用线性模型
                        model = np.polyfit(x_prior, y_prior, 1)
                        predicted_value = np.polyval(model, i)
                else:
                    # 默认使用线性外插
                    model = np.polyfit(x_prior, y_prior, 1)
                    predicted_value = np.polyval(model, i)

                # 防止异常值
                if np.isnan(predicted_value) or np.isinf(predicted_value):
                    # 使用最近的有效值
                    predicted_value = series.iloc[prior_valid_indices[-1]]

                # 更新值
                series_copy.iloc[i] = predicted_value

            except Exception as e:
                print(f"外插失败: {e}, 使用最近的有效值填充")
                # 使用最近的有效值填充
                if len(prior_valid_indices) > 0:
                    series_copy.iloc[i] = series.iloc[prior_valid_indices[-1]]
                else:
                    # 没有先前的有效值，尝试使用全局均值
                    all_valid_indices = indices[~outlier_mask]
                    if len(all_valid_indices) >= 1:
                        series_copy.iloc[i] = series.iloc[all_valid_indices].mean()
                    else:
                        series_copy.iloc[i] = 0

        return series_copy

    def _interpolate_outliers(self, series, outlier_mask):
        """插值修复异常值"""
        if outlier_mask.sum() == 0:
            return series.copy()

        series_copy = series.copy()
        valid_indices = ~outlier_mask

        if valid_indices.sum() < 2:
            # 如果有效点太少，用均值填充
            fill_value = series[valid_indices].mean()
            if pd.isna(fill_value):  # 如果均值为NaN，尝试用中位数
                fill_value = series.median()
                if pd.isna(fill_value):  # 如果中位数也为NaN，用0填充
                    fill_value = 0
            series_copy[outlier_mask] = fill_value
            return series_copy

        # 获取有效数据点
        valid_x = np.where(valid_indices)[0]
        valid_y = series[valid_indices].values

        # 需要插值的点
        outlier_x = np.where(outlier_mask)[0]

        try:
            if self.interpolation_method == 'linear':
                f = interpolate.interp1d(valid_x, valid_y, kind='linear',
                                         bounds_error=False, fill_value='extrapolate')
            elif self.interpolation_method == 'cubic' and len(valid_x) >= 4:
                f = interpolate.interp1d(valid_x, valid_y, kind='cubic',
                                         bounds_error=False, fill_value='extrapolate')
            elif self.interpolation_method == 'quadratic' and len(valid_x) >= 3:
                f = interpolate.interp1d(valid_x, valid_y, kind='quadratic',
                                         bounds_error=False, fill_value='extrapolate')
            else:
                f = interpolate.interp1d(valid_x, valid_y, kind='linear',
                                         bounds_error=False, fill_value='extrapolate')

            # 执行插值
            interpolated_values = f(outlier_x)
            series_copy.iloc[outlier_x] = interpolated_values

        except Exception as e:
            print(f"插值失败，使用均值填充: {e}")
            fill_value = series[valid_indices].mean()
            if pd.isna(fill_value):
                fill_value = 0
            series_copy[outlier_mask] = fill_value

        return series_copy

    def _process_point_outliers(self):
        """处理点迹数据异常值"""
        if self.point_data is None:
            return None

        processed_data = self.point_data.copy()

        # 按批号分组处理
        for batch_id in processed_data['批号'].unique():
            batch_mask = processed_data['批号'] == batch_id
            batch_data = processed_data[batch_mask].copy()

            if len(batch_data) < 3:
                continue

            # 处理多普勒速度异常值
            doppler_series = batch_data['多普勒速度']

            # 组合检测方法
            zscore_outliers = self._detect_outliers_zscore(doppler_series)
            velocity_outliers = self._detect_velocity_outliers(doppler_series, self.doppler_threshold)
            iqr_outliers = self._detect_outliers_iqr(doppler_series)

            # 综合异常值检测（任意一种方法检测到就认为是异常值）
            combined_outliers = zscore_outliers | velocity_outliers | iqr_outliers

            if combined_outliers.sum() > 0:
                if self.verbose:
                    print(f"批号 {batch_id}: 检测到 {combined_outliers.sum()} 个多普勒速度异常值")
                corrected_doppler = self._interpolate_outliers(doppler_series, combined_outliers)
                processed_data.loc[batch_mask, '多普勒速度'] = corrected_doppler.values

            # 处理其他可能的异常值
            for col in ['距离', '方位', '俯仰', '信噪比']:
                if col in batch_data.columns:
                    series = batch_data[col]
                    outliers = self._detect_outliers_zscore(series)
                    if outliers.sum() > 0:
                        if self.verbose:
                            print(f"批号 {batch_id}: 检测到 {outliers.sum()} 个{col}异常值")
                        # corrected_series = self._interpolate_outliers(series, outliers)
                        corrected_series = self._extrapolate_outliers(series, outliers)
                        processed_data.loc[batch_mask, col] = corrected_series.values

        return processed_data

    def _process_track_outliers(self):
        """处理航迹数据异常值"""
        if self.track_data is None:
            return None

        processed_data = self.track_data.copy()

        # 按批号分组处理
        for batch_id in processed_data['批号'].unique():
            batch_mask = processed_data['批号'] == batch_id
            batch_data = processed_data[batch_mask].copy()

            if len(batch_data) < 3:
                continue

            # 处理速度异常值
            velocity_columns = ['X向速度', 'Y向速度', 'Z向速度', '全速度']

            for col in velocity_columns:
                if col in batch_data.columns:
                    velocity_series = batch_data[col]

                    # 检测速度异常值
                    velocity_outliers = self._detect_velocity_outliers(velocity_series, self.velocity_threshold)
                    zscore_outliers = self._detect_outliers_zscore(velocity_series)

                    combined_outliers = velocity_outliers | zscore_outliers

                    if combined_outliers.sum() > 0:
                        if self.verbose:
                            print(f"批号 {batch_id}: 检测到 {combined_outliers.sum()} 个{col}异常值")
                        corrected_velocity = self._interpolate_outliers(velocity_series, combined_outliers)
                        processed_data.loc[batch_mask, col] = corrected_velocity.values

            # 处理位置和角度异常值
            for col in ['滤波距离', '滤波方位', '滤波俯仰', '航向']:
                if col in batch_data.columns:
                    series = batch_data[col]
                    outliers = self._detect_outliers_zscore(series)
                    if outliers.sum() > 0:
                        if self.verbose:
                            print(f"批号 {batch_id}: 检测到 {outliers.sum()} 个{col}异常值")
                        # corrected_series = self._interpolate_outliers(series, outliers)
                        corrected_series = self._extrapolate_outliers(series, outliers)
                        processed_data.loc[batch_mask, col] = corrected_series.values

        return processed_data

    def get_processed_data(self):
        """获取处理后的数据"""
        return {
            'point_data': self.processed_point_data,
            'track_data': self.processed_track_data
        }

    def save_processed_data(self, output_dir='processed_data'):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)

        if self.processed_point_data is not None:
            os.makedirs(os.path.join(output_dir, '点迹'), exist_ok=True)
            point_output_path = os.path.join(output_dir, '点迹', os.path.basename(self.point_file_path))
            self.processed_point_data.to_csv(point_output_path, index=False)

        if self.processed_track_data is not None:
            os.makedirs(os.path.join(output_dir, '航迹'), exist_ok=True)
            track_output_path = os.path.join(output_dir, '航迹', os.path.basename(self.track_file_path))
            self.processed_track_data.to_csv(track_output_path, index=False)


if __name__ == '__main__':
    import plotly.graph_objects as go
    from math import log10

    data_root = "D:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集"
    save_dir = "D:/DataSets/挑战杯_揭榜挂帅_CQ-08赛题_数据集/processed_data"
    point_files = glob.glob(os.path.join(data_root, "点迹", "PointTracks_*.txt"))
    dopplers = {'1': [], '2': [], '3': [], '4': []}
    for point_file in tqdm(point_files, desc="处理异常数据"):
        re_result = re.match(r"PointTracks_(\d+)_(\d+)_(\d+).txt", os.path.basename(point_file))
        batch_id = re_result.group(1)
        label = re_result.group(2)
        if int(label) > 4:
            continue
        num_points = re_result.group(3)
        track_file = os.path.join(data_root, "航迹", f"Tracks_{batch_id}_{label}_{num_points}.txt")
        preprocessor = TrajectoryDataProcessor(point_file_path=point_file, track_file_path=track_file)
        result = preprocessor.get_processed_data()
        point_df = result['point_data']
        for i in range(len(point_df)):
            try:
                doppler = log10(point_df['信噪比'][i])
                dopplers[label].append(doppler)
            except Exception as e:
                print(f"{point_file} 中信噪比异常: {point_df['时间'][i]}")
    for k, v in dopplers.items():
        fig = go.Figure(data=[go.Histogram(x=v, histnorm='probability', nbinsx=30)])
        fig.update_layout(
            title_text="信噪比分布",
            xaxis_title_text="信噪比 (对数坐标)",
            yaxis_title_text="占比",
            bargap=0.2,
            bargroupgap=0.1
        )
        fig.show()