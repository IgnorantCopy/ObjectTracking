import os
import re
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional

from .data_loader import DataLoader
from .outlier_detection import OutlierDetector
from .feature_engineering import FeatureEngineering
from .rd_preprocessor import BatchFile


class TrajectoryPreprocessor:
    """轨迹数据预处理器 - 适配深度学习模型"""
    
    def __init__(self, data_root: str, seq_len: int, num_classes: int,
                 num_features: int, test: bool = False, verbose: bool = False):
        self.data_root = data_root
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_features = num_features
        self.test = test
        self.verbose = verbose
        
        # 初始化各模块
        self.data_loader = DataLoader()
        self.outlier_detector = OutlierDetector(verbose=verbose)
        self.feature_engineer = FeatureEngineering()
        
        # 数据存储
        self.sequences = []
        self.labels = []
        self.batch_ids = []
    
    def process_single_trajectory(self, batch_file: BatchFile) -> Optional[Tuple[torch.Tensor]]:
        """
        处理单个轨迹文件对
        
        Args:
            point_file: 点迹文件路径
            track_file: 航迹文件路径
            
        Returns:
            (特征序列, 批号) 或 None（如果处理失败）
        """
        point_file = batch_file.point_file
        track_file = batch_file.track_file
        # 从文件名提取信息
        pattern = r"PointTracks_(\d+)_(\d+).txt" if self.test else r"PointTracks_(\d+)_(\d+)_(\d+).txt"
        re_result = re.match(pattern, os.path.basename(point_file))
        if not re_result:
            if self.verbose:
                print(f"无法解析文件名: {point_file}")
            return None

        batch_id = re_result.group(1)
        
        # 加载数据
        point_data, track_data = self.data_loader.load_data_pair(point_file, track_file)
        
        if point_data is None or track_data is None:
            if self.verbose:
                print(f"数据加载失败: {point_file}, {track_file}")
            return None
        
        if len(point_data) == 0 or len(track_data) == 0:
            if self.verbose:
                print(f"数据为空: {point_file}, {track_file}")
            return None
        
        try:
            # 合并点迹和航迹数据
            merged_data = self._merge_point_track_data(point_data, track_data)
            
            if merged_data is None or len(merged_data) == 0:
                if self.verbose:
                    print(f"数据合并失败: {batch_id}")
                return None
            
            # 异常值检测和修复
            cleaned_data = self._clean_data(merged_data, batch_id)
            
            # 特征工程
            features = self.feature_engineer.process_features(cleaned_data)
            
            # 序列化处理
            sequence = self._create_sequence(features)
            
            if sequence is None:
                return None

            return torch.from_numpy(sequence).float()
            
        except Exception as e:
            if self.verbose:
                print(f"处理轨迹失败 {batch_id}: {e}")
            return None

    @staticmethod
    def data_padding(merged_data, track_seq_len, N=4):
        """
        对点迹数据进行智能填充，通过寻找历史相似模式
        :param merged_data: 点迹数据 (NumPy Array)
        :param track_seq_len: 目标长度
        :param N: 用于模式匹配的行数,不超过6
        :return: 填充后的完整数据
        """
        current_len = merged_data.shape[0]
        padding_length = track_seq_len - current_len
        reference_pattern = merged_data[-N:, :]  # shape: [N, features]

        # 确定搜索范围
        if current_len >= padding_length + 2 * N:
            search_end = min(current_len - padding_length - N, current_len - 2 * N)
        else:
            search_end = current_len - 2 * N
        search_end = max(0, search_end)

        min_distance = float("inf")
        best_start_idx = N + 1

        # 遍历可能的起始位置

        from scipy.spatial.distance import cosine

        for start_idx in range(search_end):
            if start_idx + N > current_len:
                break
            current_pattern = merged_data[
                              start_idx: start_idx + N, :
                              ]  # shape: [N, features]
            total_distance = 0.0
            for i in range(N):
                row_distance_uc = np.sqrt(
                    np.sum((reference_pattern[i] - current_pattern[i]) ** 2)
                )
                row_distance_cos = cosine(reference_pattern[i], current_pattern[i])
                row_distance = (row_distance_uc + row_distance_cos) / 2
                total_distance += row_distance
            if total_distance < min_distance:
                min_distance = total_distance
                best_start_idx = start_idx + N + 1

        padding_data_list = []
        remaining_padding = padding_length
        while remaining_padding > 0:
            # 确定这次可以填充多少行
            available_rows = current_len - best_start_idx
            rows_to_add = min(remaining_padding, available_rows)

            padding_segment = merged_data[best_start_idx: best_start_idx + rows_to_add, :]
            padding_data_list.append(padding_segment)
            remaining_padding -= rows_to_add

            # 如果还需要更多数据，重新开始循环
            if remaining_padding > 0 and available_rows < remaining_padding:
                pass

        # 合并所有填充数据
        if padding_data_list:
            padding_data = np.concatenate(padding_data_list, axis=0)
            assert (
                    padding_data.shape[0] == padding_length
            ), f"填充数据行数不匹配: {padding_data.shape[0]}, 期望: {padding_length}"
        else:
            # 如果没有找到合适的模式，回退到重复最后一行
            padding_data = np.stack(
                [merged_data[-1, :] for _ in range(padding_length)], axis=0
            )
        padding_data[:, -2] = 1
        return padding_data

    @staticmethod
    def _merge_point_track_data(point_data: pd.DataFrame, track_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        合并点迹和航迹数据
        
        Args:
            point_data: 点迹数据
            track_data: 航迹数据
            
        Returns:
            合并后的数据
        """
        # 按时间和批号合并
        merged = pd.merge(track_data, point_data,  on=['时间', '批号'], how='inner')
        # 按时间排序
        merged = merged.sort_values('时间').reset_index(drop=True)
        merged['padding'] = np.zeros(len(merged))  # 填充列
        merged['类别'] = np.ones(len(merged)) * -1  # 类别列

        return merged
            
    def _clean_data(self, data: pd.DataFrame, batch_id: str) -> pd.DataFrame:
        """
        清理数据中的异常值
        
        Args:
            data: 原始合并数据
            batch_id: 批号
            
        Returns:
            清理后的数据
        """
        cleaned_data = data.copy()
        
        if len(cleaned_data) < 3:
            return cleaned_data
        
        # 处理多普勒速度异常值
        if '多普勒速度' in cleaned_data.columns:
            doppler_series = cleaned_data['多普勒速度']
            
            # 组合检测方法
            zscore_outliers = self.outlier_detector.detect_outliers_zscore(doppler_series)
            velocity_outliers = self.outlier_detector.detect_velocity_outliers(
                doppler_series, self.outlier_detector.doppler_threshold)
            iqr_outliers = self.outlier_detector.detect_outliers_iqr(doppler_series)
            
            combined_outliers = zscore_outliers | velocity_outliers | iqr_outliers
            
            if combined_outliers.sum() > 0:
                if self.verbose:
                    print(f"批号 {batch_id}: 检测到 {combined_outliers.sum()} 个多普勒速度异常值")
                corrected_doppler = self.outlier_detector.extrapolate_outliers(doppler_series, combined_outliers)
                cleaned_data['多普勒速度'] = corrected_doppler
        
        # 处理其他数值列异常值
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['时间', '批号']:
                continue
            
            series = cleaned_data[col]
            outliers = self.outlier_detector.detect_outliers_zscore(series)
            
            if outliers.sum() > 0:
                if self.verbose:
                    print(f"批号 {batch_id}: 检测到 {outliers.sum()} 个{col}异常值")
                
                # 对位置和角度使用外插，对其他使用插值
                if col in ['滤波距离', '滤波方位', '滤波俯仰', '航向', '距离', '方位', '俯仰']:
                    corrected_series = self.outlier_detector.extrapolate_outliers(series, outliers)
                else:
                    corrected_series = self.outlier_detector.extrapolate_outliers(series, outliers)
                
                cleaned_data[col] = corrected_series
        
        return cleaned_data
    
    def _create_sequence(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """
        创建固定长度的序列
        
        Args:
            features: 特征数据
            
        Returns:
            形状为 (seq_len, num_features) 的序列数组
        """
        if len(features) == 0:
            return None
        
        # 确保特征数量正确
        if len(features.columns) != self.num_features:
            if self.verbose:
                print(f"特征数量不匹配: 期望 {self.num_features}, 实际 {len(features.columns)}")
            return None
        
        feature_array = features.values.astype(np.float32)
        
        # 处理序列长度
        if len(feature_array) >= self.seq_len:
            # 如果序列太长，截取前seq_len个时间步
            sequence = feature_array[:self.seq_len]
        else:
            # 如果序列太短，进行填充
            sequence = np.zeros((self.seq_len, self.num_features), dtype=np.float32)
            sequence[:len(feature_array)] = feature_array
            
            # 用最后一个有效值填充剩余部分
            if len(feature_array) > 0:
                last_valid = feature_array[-1]
                sequence[len(feature_array):] = last_valid
                sequence[len(feature_array):, -2] = 1  # 标记为 padding
        
        return sequence
    