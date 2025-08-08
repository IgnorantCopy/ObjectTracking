"""
特征工程模块
"""
import pandas as pd
import polars as pl
from ensemble.track.configs.config import FINAL_FEATURE_COLUMNS


class FeatureEngineering:
    """特征工程类"""
    
    @staticmethod
    def process_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        对合并的点迹+航迹数据进行特征工程
        
        Args:
            df: 包含点迹和航迹数据的DataFrame
            
        Returns:
            处理后的特征DataFrame
        """
        # 转换为polars以提高性能（保持与原脚本一致）
        df_pl = pl.from_pandas(df)
        
        # 应用原始的特征工程逻辑
        df_processed = FeatureEngineering._process_df_polars(df_pl)
        
        # 转回pandas
        return df_processed.to_pandas()
    
    @staticmethod
    def _process_df_polars(df: pl.DataFrame) -> pl.DataFrame:
        """
        使用polars进行特征工程（直接移植自track_preprocess.py）
        """
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
            (pl.col("X向速度").pow(2) + pl.col("Y向速度").pow(2) + pl.col("Z向速度").pow(2)).sqrt().alias(
                "三维速度模长"),
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
        
        # 选择最终特征
        df_final_features = df.select(FINAL_FEATURE_COLUMNS)
        
        # 填充所有因计算差分等产生的空值
        df_final_features = df_final_features.fill_null(0.0).fill_nan(0.0)

        return df_final_features