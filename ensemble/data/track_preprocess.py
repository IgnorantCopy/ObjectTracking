import os
import re
import glob
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import zscore
from tqdm import tqdm


def process_df(df: pl.DataFrame) -> pl.DataFrame:
    # 特征工程
    df = df.with_columns(
        (pl.col("滤波距离") * pl.col("滤波俯仰").radians().sin()).alias("高度"),
        (pl.col("X向速度").pow(2) + pl.col("Y向速度").pow(2)).sqrt().alias("水平速度"),
    )
    df = df.with_columns(
        pl.arctan2(pl.col("Z向速度"), pl.col("水平速度")).alias("爬升/俯冲角度_弧度"),
        (pl.col("和幅度") * pl.col("滤波距离").pow(4)).log10().alias("RCS"),
    )
    df = df.with_columns(
        # 当前时刻的速度向量模长
        (pl.col("X向速度").pow(2) + pl.col("Y向速度").pow(2) + pl.col("Z向速度").pow(2)).sqrt().alias("三维速度模长"),
        # 计算时间差（用于角速度计算）
        (pl.col("时间").diff().dt.total_seconds()).alias("时间差"),
    )
    df = df.with_columns(
        # 相邻时刻速度向量点积
        (pl.col("X向速度") * pl.col("X向速度").shift(1) +
         pl.col("Y向速度") * pl.col("Y向速度").shift(1) +
         pl.col("Z向速度") * pl.col("Z向速度").shift(1)).alias("速度向量点积"),
        # 相邻时刻速度向量模长乘积
        (pl.col("三维速度模长") * pl.col("三维速度模长").shift(1)).alias("速度模长乘积"),
    )
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
    df = df.with_columns(
        (pl.col("高度").cum_max() - pl.col("高度").cum_min()).alias("高度波动范围"),
        (pl.col("水平速度").cum_max() - pl.col("水平速度").cum_min()).alias("水平速度波动范围"),
        # 累积标准差（使用正确的方差公式）
        (pl.col("高度").pow(2).cum_sum() / pl.col("高度").cum_count() -
         (pl.col("高度").cum_sum() / pl.col("高度").cum_count()).pow(2)).sqrt().alias("高度标准差"),
        # 转向角度和角速度的累积标准差
        (pl.col("转向角度").pow(2).cum_sum() / pl.col("转向角度").cum_count() -
         (pl.col("转向角度").cum_sum() / pl.col("转向角度").cum_count()).pow(2)).sqrt().alias("转向角度累积标准差"),
        (pl.col("角速度").pow(2).cum_sum() / pl.col("角速度").cum_count() -
         (pl.col("角速度").cum_sum() / pl.col("角速度").cum_count()).pow(2)).sqrt().alias("角速度累积标准差"),
        # 抖动指数（二阶差分的RMS）
        (pl.col("多普勒二阶差分").pow(2).cum_sum() / pl.col("多普勒二阶差分").cum_count()).sqrt().alias("多普勒抖动指数"),
        (pl.col("幅度二阶差分").pow(2).cum_sum() / pl.col("幅度二阶差分").cum_count()).sqrt().alias("幅度抖动指数"),
        ((pl.col("距离二阶差分").pow(2) + pl.col("方位二阶差分").pow(2) + pl.col("俯仰二阶差分").pow(2))
         .cum_sum() / pl.col("距离二阶差分").cum_count()).sqrt().alias("位置抖动指数"),
    )
    # 最终的特征
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
    # 填充所有因计算差分等产生的空值
    df_final_features = df_final_features.fill_null(0.0).fill_nan(0.0)

    return df_final_features


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
                corrected_doppler = self._extrapolate_outliers(doppler_series, combined_outliers)
                processed_data.loc[batch_mask, '多普勒速度'] = corrected_doppler.values

            # 处理其他可能的异常值
            for col in ['距离', '方位', '俯仰', '信噪比']:
                if col in batch_data.columns:
                    series = batch_data[col]
                    outliers = self._detect_outliers_zscore(series)
                    if outliers.sum() > 0:
                        if self.verbose:
                            print(f"批号 {batch_id}: 检测到 {outliers.sum()} 个{col}异常值")
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
                        corrected_velocity = self._extrapolate_outliers(velocity_series, combined_outliers)
                        processed_data.loc[batch_mask, col] = corrected_velocity.values

            # 处理位置和角度异常值
            for col in ['滤波距离', '滤波方位', '滤波俯仰', '航向']:
                if col in batch_data.columns:
                    series = batch_data[col]
                    outliers = self._detect_outliers_zscore(series)
                    if outliers.sum() > 0:
                        if self.verbose:
                            print(f"批号 {batch_id}: 检测到 {outliers.sum()} 个{col}异常值")
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