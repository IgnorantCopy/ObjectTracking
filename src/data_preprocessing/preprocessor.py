"""
主预处理器
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm

from .data_loader import DataLoader
from .outlier_detection import OutlierDetector
from .feature_engineering import FeatureEngineering
from ..utils.config import (
    DATA_ROOT, NUM_CLASSES, TOTAL_FEATURES_PER_TIMESTEP, SEQ_LEN,
    ABNORMAL_BATCH_ID, FINAL_FEATURE_COLUMNS
)


class TrajectoryPreprocessor:
    """轨迹数据预处理器 - 适配深度学习模型"""
    
    def __init__(self, 
                 data_root: str = DATA_ROOT,
                 seq_len: int = SEQ_LEN,
                 num_classes: int = NUM_CLASSES,
                 test: bool = False,
                 verbose: bool = False):
        self.data_root = data_root
        self.seq_len = seq_len
        self.num_classes = num_classes
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
    
    def process_single_trajectory(self, point_file: str, track_file: str) -> Optional[Tuple[np.ndarray, int, str]]:
        """
        处理单个轨迹文件对
        
        Args:
            point_file: 点迹文件路径
            track_file: 航迹文件路径
            
        Returns:
            (特征序列, 标签, 批号) 或 None（如果处理失败）
        """
        # 从文件名提取信息
        pattern = r"PointTracks_(\d+)_(\d+).txt" if self.test else r"PointTracks_(\d+)_(\d+)_(\d+).txt"
        re_result = re.match(pattern, os.path.basename(point_file))
        if not re_result:
            if self.verbose:
                print(f"无法解析文件名: {point_file}")
            return None

        batch_id = re_result.group(1)
        if not self.test:
            label = int(re_result.group(2))

            if self.verbose:
                print(f"处理批号 {batch_id}, 标签 {label}")

            # 只处理1-4类标签
            if label > 4:
                if self.verbose:
                    print(f"跳过标签 {label} (>4)")
                return None
        
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
            if self.test:
                return sequence, -1, batch_id  # 测试集无标签
            return sequence, label - 1, batch_id  # 标签转换为0-3
            
        except Exception as e:
            if self.verbose:
                print(f"处理轨迹失败 {batch_id}: {e}")
            return None
    
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
        if len(features.columns) != TOTAL_FEATURES_PER_TIMESTEP:
            if self.verbose:
                print(f"特征数量不匹配: 期望 {TOTAL_FEATURES_PER_TIMESTEP}, 实际 {len(features.columns)}")
            return None
        
        feature_array = features.values.astype(np.float32)
        
        # 处理序列长度
        if len(feature_array) >= self.seq_len:
            # 如果序列太长，截取前seq_len个时间步
            sequence = feature_array[:self.seq_len]
        else:
            # 如果序列太短，进行填充
            sequence = np.zeros((self.seq_len, TOTAL_FEATURES_PER_TIMESTEP), dtype=np.float32)
            sequence[:len(feature_array)] = feature_array
            
            # 用最后一个有效值填充剩余部分
            if len(feature_array) > 0:
                last_valid = feature_array[-1]
                sequence[len(feature_array):] = last_valid
                sequence[len(feature_array):, -2] = 1  # 标记为 padding
        
        return sequence
    
    def process_all_trajectories(self) -> Dict[str, Any]:
        """
        处理所有轨迹数据
        
        Returns:
            包含处理结果的字典
        """
        point_files = glob.glob(os.path.join(self.data_root, "点迹", "PointTracks_*.txt"))
        
        success_count = 0
        fail_count = 0
        
        self.sequences = []
        self.labels = []
        self.batch_ids = []
        
        for point_file in tqdm(point_files, desc="处理轨迹数据"):
            pattern = r"PointTracks_(\d+)_(\d+).txt" if self.test else r"PointTracks_(\d+)_(\d+)_(\d+).txt"
            re_result = re.match(pattern, os.path.basename(point_file))
            if not re_result:
                fail_count += 1
                continue
            
            batch_id = re_result.group(1)
            if batch_id in ABNORMAL_BATCH_ID:
                continue
            if self.test:
                num_points = re_result.group(2)
                track_file = os.path.join(self.data_root, "航迹", f"Tracks_{batch_id}_{num_points}.txt")
            else:
                label = re_result.group(2)
                num_points = re_result.group(3)
                track_file = os.path.join(self.data_root, "航迹", f"Tracks_{batch_id}_{label}_{num_points}.txt")
            
            result = self.process_single_trajectory(point_file, track_file)
            
            if result is not None:
                sequence, label_idx, batch_id = result
                self.sequences.append(sequence)
                self.labels.append(label_idx)
                self.batch_ids.append(batch_id)
                success_count += 1
            else:
                fail_count += 1
        
        if self.verbose:
            print(f"处理完成: 成功 {success_count}, 失败 {fail_count}")
        
        return {
            'sequences': np.array(self.sequences) if self.sequences else np.array([]),
            'labels': np.array(self.labels) if self.labels else np.array([]),
            'batch_ids': self.batch_ids,
            'success_count': success_count,
            'fail_count': fail_count
        }
    
    def get_pytorch_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取PyTorch格式的数据集
        
        Returns:
            (特征张量, 标签张量)
        """
        if not self.sequences or not self.labels:
            raise ValueError("请先调用 process_all_trajectories() 处理数据")
        
        X = torch.from_numpy(np.array(self.sequences)).float()
        y = torch.from_numpy(np.array(self.labels)).long()
        
        return X, y
    
    def save_processed_data(self, output_path: str):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出路径
        """
        if not self.sequences or not self.labels:
            raise ValueError("没有数据可保存")
        
        np.savez_compressed(
            output_path,
            sequences=np.array(self.sequences),
            labels=np.array(self.labels),
            batch_ids=self.batch_ids,
            seq_len=self.seq_len,
            num_features=TOTAL_FEATURES_PER_TIMESTEP,
            num_classes=self.num_classes
        )
        
        if self.verbose:
            print(f"数据已保存到: {output_path}")
    
    def load_processed_data(self, input_path: str) -> Dict[str, Any]:
        """
        加载处理后的数据
        
        Args:
            input_path: 输入路径
            
        Returns:
            加载的数据字典
        """
        data = np.load(input_path, allow_pickle=True)
        
        self.sequences = data['sequences'].tolist()
        self.labels = data['labels'].tolist()
        self.batch_ids = data['batch_ids'].tolist()
        self.seq_len = int(data['seq_len'])
        
        if self.verbose:
            print(f"数据已从 {input_path} 加载")
            print(f"序列数量: {len(self.sequences)}")
            print(f"序列长度: {self.seq_len}")
            print(f"特征维度: {data['num_features']}")
        
        return {
            'sequences': data['sequences'],
            'labels': data['labels'],
            'batch_ids': data['batch_ids'],
            'seq_len': data['seq_len'],
            'num_features': data['num_features'],
            'num_classes': data['num_classes']
        }