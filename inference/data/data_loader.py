"""
数据加载模块
"""
import os
import pandas as pd
from typing import Optional, Tuple
from ensemble.track.configs.config import POINT_COLUMNS, TRACK_COLUMNS


class DataLoader:
    """数据加载器"""
    
    def __init__(self, encoding='gbk'):
        self.encoding = encoding
    
    def load_point_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        加载点迹数据
        
        Args:
            file_path: 点迹文件路径
            
        Returns:
            点迹数据DataFrame，加载失败返回None
        """
        try:
            data = pd.read_csv(file_path, encoding=self.encoding, header=0, names=POINT_COLUMNS)
            # 转换时间格式
            data['时间'] = pd.to_datetime(data['时间'], format='%H:%M:%S.%f')
            return data
        except Exception as e:
            print(f"加载点迹数据失败 {file_path}: {e}")
            return None
    
    def load_track_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        加载航迹数据
        
        Args:
            file_path: 航迹文件路径
            
        Returns:
            航迹数据DataFrame，加载失败返回None
        """
        try:
            data = pd.read_csv(file_path, encoding=self.encoding, header=0, names=TRACK_COLUMNS)
            # 转换时间格式
            data['时间'] = pd.to_datetime(data['时间'], format='%H:%M:%S.%f')
            return data
        except Exception as e:
            print(f"加载航迹数据失败 {file_path}: {e}")
            return None
    
    def load_data_pair(self, point_file: str, track_file: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        加载点迹和航迹数据对
        
        Args:
            point_file: 点迹文件路径
            track_file: 航迹文件路径
            
        Returns:
            (点迹数据, 航迹数据) 元组
        """
        point_data = self.load_point_data(point_file) if point_file and os.path.exists(point_file) else None
        track_data = self.load_track_data(track_file) if track_file and os.path.exists(track_file) else None
        
        return point_data, track_data