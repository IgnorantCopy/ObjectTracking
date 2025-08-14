"""
流式推理配置文件
"""
from dataclasses import dataclass


@dataclass
class StreamingConfig:
    """流式推理配置"""
    
    # 模型参数
    # min_seq_len: int = 10  # 最小序列长度
    num_features: int = 10_000  # 特征数
    dropout: float = 0.2  # dropout概率
    confidence_threshold: float = 0.95  # 置信度阈值
    
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-2
    weight_decay: float = 5e-4
    epochs: int = 100
    early_stopping_patience: int = 20
    
    # 数据分割
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle: bool = True
    seed: int = 42
    
    # 设备设置
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # 保存设置
    save_dir: str = "../checkpoints"
    save_best_only: bool = True
    
    # 日志设置
    log_level: str = "INFO"
    
    def __post_init__(self):
        """验证配置参数"""
        assert self.train_split + self.val_split + self.test_split == 1.0, "数据分割比例必须总和为1"
        assert 0 < self.confidence_threshold < 1, "置信度阈值必须在0-1之间"
        # assert self.min_seq_len >= 10, "最小序列长度必须>=10（HydraMultiRocket限制）"


@dataclass 
class EvaluationConfig:
    """评估配置"""
    
    # 评估设置
    detailed_analysis: bool = True
    max_eval_samples: int = 500  # 限制评估样本数以节省时间
    streaming_analysis: bool = True
    
    # 可视化设置
    plot_results: bool = True
    save_plots: bool = True
    plot_dpi: int = 150
    
    # 报告设置
    generate_report: bool = True
    report_format: str = "markdown"  # "markdown", "html", "pdf"