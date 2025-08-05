"""
训练配置文件
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 数据配置
    data_path: str = "processed_data.npz"
    batch_size: int = 32
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle: bool = True
    num_workers: int = 4
    
    # 模型配置
    seq_len: int = 20
    input_features: int = 28
    num_classes: int = 4
    num_features: int = 20000
    dropout: float = 0.2
    model_kwargs: dict = None
    
    # 训练配置
    epochs: int = 100
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20
    
    # 优化器配置
    optimizer: str = "AdamW"  # AdamW, Adam, SGD
    momentum: float = 0.9  # for SGD
    
    # 损失函数配置
    loss_function: str = "CrossEntropyLoss"
    label_smoothing: float = 0.0
    
    # 设备配置
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = False  # 暂时禁用混合精度
    
    # 保存配置
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    save_last: bool = True
    
    # 日志配置
    log_dir: str = "logs"
    log_interval: int = 10
    val_interval: int = 1
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        """后处理配置"""
        if self.model_kwargs is None:
            self.model_kwargs = {}
        
        # 自动检测设备
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 验证分割比例
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "train_split + val_split + test_split must equal 1.0"


# 预定义配置
def get_default_config() -> TrainingConfig:
    """获取默认配置"""
    return TrainingConfig()


def get_fast_config() -> TrainingConfig:
    """获取快速训练配置（用于测试）"""
    return TrainingConfig(
        epochs=10,
        batch_size=64,
        early_stopping_patience=5,
        scheduler_patience=3
    )


def get_stable_config() -> TrainingConfig:
    """获取稳定训练配置（较小批次大小）"""
    return TrainingConfig(
        batch_size=64,
        epochs=100,
        learning_rate=1e-2,
        scheduler_patience=10,
        early_stopping_patience=20,
        weight_decay=5e-4
    )