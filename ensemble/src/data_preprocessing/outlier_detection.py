"""
异常值检测模块
"""
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy import interpolate
from typing import Union
from ..utils.config import OUTLIER_THRESHOLD, INTERPOLATION_METHOD, VELOCITY_THRESHOLD, DOPPLER_THRESHOLD


class OutlierDetector:
    """异常值检测器"""
    
    def __init__(self, 
                 outlier_threshold: float = OUTLIER_THRESHOLD,
                 interpolation_method: str = INTERPOLATION_METHOD,
                 velocity_threshold: float = VELOCITY_THRESHOLD,
                 doppler_threshold: float = DOPPLER_THRESHOLD,
                 verbose: bool = False):
        self.outlier_threshold = outlier_threshold
        self.interpolation_method = interpolation_method
        self.velocity_threshold = velocity_threshold
        self.doppler_threshold = doppler_threshold
        self.verbose = verbose
    
    def detect_outliers_zscore(self, series: pd.Series, threshold: float = None) -> pd.Series:
        """使用Z-score检测异常值"""
        if threshold is None:
            threshold = self.outlier_threshold
        
        if len(series) <= 1:
            return pd.Series([False] * len(series), index=series.index)
        
        z_scores = np.abs(zscore(series, nan_policy='omit'))
        return pd.Series(z_scores > threshold, index=series.index)
    
    def detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """使用IQR方法检测异常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_velocity_outliers(self, series: pd.Series, threshold: float) -> pd.Series:
        """检测速度异常值（绝对值过大）"""
        return np.abs(series) > threshold
    
    def extrapolate_outliers(self, series: pd.Series, outlier_mask: pd.Series) -> pd.Series:
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
                    fill_value = series.iloc[all_valid_indices].mean()
                    if pd.isna(fill_value):
                        fill_value = 0
                    series_copy.iloc[i] = fill_value
                else:
                    series_copy.iloc[i] = 0
                continue
            
            # 获取之前的有效点的数据
            x_prior = prior_valid_indices
            y_prior = series.iloc[prior_valid_indices].values
            
            try:
                # 线性外插
                if self.interpolation_method == 'linear':
                    model = np.polyfit(x_prior, y_prior, 1)
                    predicted_value = np.polyval(model, i)
                elif self.interpolation_method == 'quadratic' and len(x_prior) >= 3:
                    model = np.polyfit(x_prior, y_prior, 2)
                    predicted_value = np.polyval(model, i)
                else:
                    model = np.polyfit(x_prior, y_prior, 1)
                    predicted_value = np.polyval(model, i)
                
                # 防止异常值
                if np.isnan(predicted_value) or np.isinf(predicted_value):
                    predicted_value = series.iloc[prior_valid_indices[-1]]
                
                series_copy.iloc[i] = predicted_value
                
            except Exception as e:
                if self.verbose:
                    print(f"外插失败: {e}, 使用最近的有效值填充")
                if len(prior_valid_indices) > 0:
                    series_copy.iloc[i] = series.iloc[prior_valid_indices[-1]]
                else:
                    series_copy.iloc[i] = 0
        
        return series_copy