"""
配置文件
"""
# 最终特征列名（基于track_preprocess.py）
FINAL_FEATURE_COLUMNS = [
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
    "俯仰", "多普勒速度", "和幅度", "信噪比", "原始点数量", "padding", "类别"
]

# 数据路径配置
DATA_ROOT = r"D:\DataSets\挑战杯_揭榜挂帅_CQ-08赛题_数据集\val_set"
SAVE_DIR = "./processed_data"

# 模型配置
NUM_CLASSES = 4  # 只区分1-4类
TOTAL_FEATURES_PER_TIMESTEP = len(FINAL_FEATURE_COLUMNS)
SEQ_LEN = 29  # 序列长度，可根据数据调整
ABNORMAL_BATCH_ID = [234, 1451, 1452, 1457, 1462, 1467, 1469, 1473, 1478, 1484, 1487, 1488, 1490, 1494, 1496, 1497, 1500]

# 异常值检测配置
OUTLIER_THRESHOLD = 3.0
INTERPOLATION_METHOD = 'linear'
VELOCITY_THRESHOLD = 100.0
DOPPLER_THRESHOLD = 50.0

# 数据列名
POINT_COLUMNS = ['时间', '批号', '距离', '方位', '俯仰', '多普勒速度', '和幅度', '信噪比', '原始点数量']
TRACK_COLUMNS = ['时间', '批号', '滤波距离', '滤波方位', '滤波俯仰', '全速度', 'X向速度', 'Y向速度', 'Z向速度', '航向']


