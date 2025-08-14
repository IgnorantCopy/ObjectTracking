import os
import struct
import torch
import numpy as np
import polars as pl
from scipy import signal
from scipy.fft import fft, fftshift
from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms


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

        # 最终判定
        if is_target:
            status_map[local_row, local_col] = 1
            output_mask[i, j] = True
        else:
            status_map[local_row, local_col] = 2

    return np.any(output_mask), output_mask, i, j


def MDB_RG_detector_2d(rd_matrix, velocity_axis,
                       detection_area_rows, detection_area_cols,
                       select_threshold_ratio=0.1, expand_threshold_ratio=0.35,
                       expand_distance=1, min_region_size=3):
    """
    对RD矩阵执行基于相对阈值的2D-MDB-RG检测算法

    :param rd_matrix: 输入的 RD 矩阵 (linear)
    :param velocity_axis: 多普勒速度轴
    :param detection_area_rows: 距离维检测区域行索引范围
    :param detection_area_cols: 多普勒维检测区域列索引范围
    :param select_threshold_ratio: 选择种子点的阈值比例 (0-1之间)
    :param expand_threshold_ratio: 区域生长的阈值比例 (0-1之间)
    :param expand_distance: 扩展距离, 默认为1(是3x3的区域)
    :param min_region_size: 最小区域大小，小于此值的区域将被忽略
    :return: 如果找到目标, 返回加了目标掩码的RD矩阵; 否则返回None
    """
    if rd_matrix.size == 0:
        return None

    rows, cols = rd_matrix.shape

    # 确保索引在有效范围内
    if len(detection_area_rows) == 0 or len(detection_area_cols) == 0:
        return None
    if detection_area_rows[0] < 0 or detection_area_rows[-1] >= rows or detection_area_cols[0] < 0 or \
            detection_area_cols[-1] >= cols:
        return None

    # 提取检测区域的数据
    detection_area_data = rd_matrix[np.ix_(detection_area_rows, detection_area_cols)]

    # 找到检测区域的中心点
    center_row_idx = len(detection_area_rows) // 2
    center_col_idx = len(detection_area_cols) // 2
    center_row = detection_area_rows[center_row_idx]
    center_col = detection_area_cols[center_col_idx]
    center_value = rd_matrix[center_row, center_col]

    # 确保中心值为正值以避免除零错误
    if center_value <= 0:
        return None

    # 获取检测区域内所有点的索引，并按幅度降序排列
    flat_indices = np.argsort(detection_area_data.ravel())[::-1]  # 降序
    rows_in_detection_area, cols_in_detection_area = np.unravel_index(
        flat_indices, detection_area_data.shape
    )

    # 寻找种子点（从高幅度开始遍历）
    seed_row = None
    seed_col = None

    for k in range(len(flat_indices)):
        curr_row_in_area = rows_in_detection_area[k]
        curr_col_in_area = cols_in_detection_area[k]

        # 将检测区域内的坐标转换为全局RD图的坐标
        i = detection_area_rows[curr_row_in_area]
        j = detection_area_cols[curr_col_in_area]

        curr_value = rd_matrix[i, j]
        if curr_value <= 0:
            continue  # 跳过非正值

        # 使用相对阈值：当前点与中心点的相对差异
        relative_diff = abs(curr_value - center_value) / max(center_value, curr_value)

        # 判断当前点与中心点的强度相对差距是否小于阈值比例
        if relative_diff < select_threshold_ratio:
            # 检查该点的速度是否超过56
            if abs(velocity_axis[j]) > 56:
                continue  # 速度过大，跳过该点

            seed_row = i
            seed_col = j
            break

    # 如果没找到合适的种子点，返回None
    if seed_row is None:
        return None

    # === 区域生长算法 ===
    # 创建掩码矩阵，标记目标区域
    target_mask = np.zeros_like(rd_matrix, dtype=bool)

    # 使用队列进行区域生长
    queue = [(seed_row, seed_col)]
    target_mask[seed_row, seed_col] = True
    seed_value = rd_matrix[seed_row, seed_col]

    while queue:
        curr_row, curr_col = queue.pop(0)
        curr_value = rd_matrix[curr_row, curr_col]

        # 定义扩展区域范围
        row_start = max(0, curr_row - expand_distance)
        row_end = min(rows, curr_row + expand_distance + 1)
        col_start = max(0, curr_col - expand_distance)
        col_end = min(cols, curr_col + expand_distance + 1)

        # 遍历扩展区域内的所有点
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                # 跳过已标记为目标的点
                if target_mask[r, c]:
                    continue

                # 获取待检测点的值
                neighbor_value = rd_matrix[r, c]

                if neighbor_value <= 0:
                    continue  # 跳过非正值

                # 计算相对差异：使用两种方法并取最小值以增强鲁棒性
                # 1. 相对于当前点的差异
                rel_diff_curr = abs(neighbor_value - curr_value) / max(neighbor_value, curr_value)
                # 2. 相对于种子点的差异
                rel_diff_seed = abs(neighbor_value - seed_value) / max(neighbor_value, seed_value)

                # 取较小的相对差异
                relative_diff = min(rel_diff_curr, rel_diff_seed)

                # 判断相邻点是否应该被纳入目标区域
                if relative_diff < expand_threshold_ratio:
                    target_mask[r, c] = True
                    queue.append((r, c))

    # 判断检测到的目标区域大小是否满足最小要求
    if np.sum(target_mask) < min_region_size:
        return None

    # 应用掩码到RD矩阵并找到目标的中心位置
    if np.any(target_mask):
        return rd_matrix * target_mask

    return None


def load_rd_map(batch: BatchFile, image_transform=None):
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
    num_points = int(os.path.basename(batch.point_file).split('_')[-1].split('.')[0])

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
            if image_transform:
                rd_matrix = image_transform(rd_matrix)
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


class RDPreprocessor(object):
    def __init__(self, image_shape, seq_len: int):
        self.channels, self.height, self.width = image_shape
        self.seq_len = seq_len
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5 for _ in range(self.channels)],
                                std=[0.5 for _ in range(self.channels)]),
        ])
    
    def preprocess(self, batch_file: BatchFile):
        images, point_index, extra_features, missing_rate = load_rd_map(batch_file, self.image_transform)
        if len(images) == 0:
            return None, None, None, None, None

        image_mask = np.ones((self.seq_len,), dtype=np.int32)
        if images.shape[0] < self.seq_len:
            image_mask[images.shape[0]:] = 0
            images = np.concatenate([
                images,
                np.zeros((self.seq_len - images.shape[0], *images.shape[1:]))
            ], axis=0)
            point_index = np.concatenate([
                point_index,
                np.array([point_index[-1] for _ in range(self.seq_len - point_index.shape[0])])
            ], axis=0)
            extra_features = np.concatenate([
                extra_features,
                np.ones((self.seq_len - extra_features.shape[0], extra_features.shape[1])) * np.mean(extra_features)
            ], axis=0)
            missing_rate = np.concatenate([
                missing_rate,
                np.array([missing_rate[-1] for _ in range(self.seq_len - missing_rate.shape[0])])
            ], axis=0)
        elif images.shape[0] > self.seq_len:
            quantiles = np.linspace(0, 1, self.seq_len)
            indices = np.floor(quantiles * (images.shape[0] - 1)).astype(int)
            images = images[indices]
            point_index = point_index[indices]
            extra_features = extra_features[indices]
            missing_rate = missing_rate[indices]
        assert images.shape[0] == self.seq_len, f"RD 图数量与预期不符: {images.shape[0]}, {self.seq_len}"

        point_index = torch.from_numpy(point_index)
        images = torch.from_numpy(images).float()
        extra_features = torch.from_numpy(extra_features).float()
        missing_rate = torch.from_numpy(missing_rate).float()
        image_mask = torch.from_numpy(image_mask)
        
        return point_index, images, extra_features, missing_rate, image_mask

