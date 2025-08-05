"""
训练器类
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

# 导入tsai模型
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from .multi_rocket_model import MultiRocketPlusClassifier

from .config import TrainingConfig
from .data_loader import TrajectoryDataLoader


class EarlyStopping:
    """早停类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """保存最佳权重"""
        self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.losses = []
        self.accuracies = []
        self.predictions = []
        self.targets = []
    
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor):
        """更新指标"""
        self.losses.append(loss)
        
        # 计算准确率 - 使用detach()切断梯度连接
        with torch.no_grad():
            pred_classes = torch.argmax(predictions.detach(), dim=1)
            accuracy = (pred_classes == targets).float().mean().item()
            self.accuracies.append(accuracy)
            
            # 保存预测和目标
            self.predictions.extend(pred_classes.cpu().numpy())
            self.targets.extend(targets.cpu().numpy())
    
    def get_average_loss(self) -> float:
        """获取平均损失"""
        return np.mean(self.losses) if self.losses else 0.0
    
    def get_average_accuracy(self) -> float:
        """获取平均准确率"""
        return np.mean(self.accuracies) if self.accuracies else 0.0
    
    def get_classification_metrics(self) -> Dict[str, Any]:
        """获取分类指标"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 计算各种指标
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
        cm = confusion_matrix(targets, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(targets, predictions)
        }


class TrajectoryTrainer:
    """轨迹分类训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # 自动检测设备
        if config.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = config.device
            
        self.device = torch.device(device_str)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 创建保存目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_loader = None
        self.early_stopping = None
        
        # 训练历史
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        
        self.logger.info(f"训练器初始化完成，使用设备: {self.device}")
        
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_data(self, data_path: Optional[str] = None):
        """设置数据加载器"""
        data_path = data_path or self.config.data_path
        
        self.data_loader = TrajectoryDataLoader(
            data_path=data_path,
            batch_size=self.config.batch_size,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            random_state=self.config.seed
        )
        
        # 更新配置中的数据维度
        data_info = self.data_loader.get_data_info()
        self.config.seq_len = data_info['seq_len']
        self.config.input_features = data_info['num_features']
        self.config.num_classes = data_info['num_classes']
        
        self.logger.info("数据加载器设置完成")
        
    def setup_model(self):
        """设置模型"""
        self.model = MultiRocketPlusClassifier(
            c_in=self.config.input_features,
            c_out=self.config.num_classes,
            seq_len=self.config.seq_len,
            num_features=self.config.num_features,
            dropout=self.config.dropout,
            **self.config.model_kwargs
        ).to(self.device)
        
        # 确保所有参数都需要梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型设置完成:")
        self.logger.info(f"  总参数量: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        
        if trainable_params == 0:
            self.logger.error("警告: 没有可训练参数！")
            # 尝试启用训练模式
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad_(True)
        
    def setup_optimizer(self):
        """设置优化器"""
        if self.config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
        
        self.logger.info(f"优化器设置完成: {self.config.optimizer}")
        
    def setup_scheduler(self):
        """设置学习率调度器"""
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience
        )
        
        self.logger.info("学习率调度器设置完成")
        
    def setup_criterion(self):
        """设置损失函数"""
        if self.config.loss_function == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        else:
            raise ValueError(f"不支持的损失函数: {self.config.loss_function}")
        
        self.logger.info(f"损失函数设置完成: {self.config.loss_function}")
        
    def setup_early_stopping(self):
        """设置早停"""
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            restore_best_weights=True
        )
        
        self.logger.info(f"早停设置完成，耐心度: {self.config.early_stopping_patience}")
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        successful_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 获取数据
            sequences = batch['sequences'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 转换维度：从 (batch, seq_len, features) 到 (batch, features, seq_len)
            sequences = sequences.transpose(1, 2)
            
            # 前向传播
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
                successful_batches += 1
            
            # 更新进度条
            current_acc = correct / labels.size(0)
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
            
            # 清理
            del loss, outputs, sequences, labels
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_acc = total_correct / total_samples
        else:
            avg_loss = 0.0
            avg_acc = 0.0
        
        return avg_loss, avg_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, Any]]:
        """验证一个epoch"""
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                sequences = batch['sequences'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 转换维度：从 (batch, seq_len, features) 到 (batch, features, seq_len)
                sequences = sequences.transpose(1, 2)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                metrics.update(loss.item(), outputs, labels)
        
        classification_metrics = metrics.get_classification_metrics()
        
        return metrics.get_average_loss(), metrics.get_average_accuracy(), classification_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        # 保存最新检查点
        if self.config.save_last:
            torch.save(checkpoint, self.save_dir / 'last_checkpoint.pth')
        
        # 保存最佳检查点
        if is_best and self.config.save_best_only:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            
        self.logger.info(f"检查点已保存 (epoch {epoch})")
    
    def train(self, data_path: Optional[str] = None):
        """开始训练"""
        self.logger.info("开始训练...")
        
        # 设置所有组件
        self.setup_data(data_path)
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_criterion()
        self.setup_early_stopping()
        
        # 获取数据加载器
        train_loader, val_loader, _ = self.data_loader.get_dataloaders()
        
        # 训练循环
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            if epoch % self.config.val_interval == 0:
                val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
                
                # 更新学习率
                self.scheduler.step(val_loss)
                
                # 记录历史
                self.train_history['loss'].append(train_loss)
                self.train_history['accuracy'].append(train_acc)
                self.val_history['loss'].append(val_loss)
                self.val_history['accuracy'].append(val_acc)
                
                # 记录日志
                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # 保存检查点
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                self.save_checkpoint(epoch, is_best)
                
                # 早停检查
                if self.early_stopping(val_loss, self.model):
                    self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                    break
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
        
        return self.train_history, self.val_history
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """评估模型"""
        if test_loader is None:
            _, _, test_loader = self.data_loader.get_dataloaders()
        
        self.logger.info("开始评估...")
        
        test_loss, test_acc, test_metrics = self.validate_epoch(test_loader)
        
        self.logger.info(f"测试结果:")
        self.logger.info(f"  测试损失: {test_loss:.4f}")
        self.logger.info(f"  测试准确率: {test_acc:.4f}")
        self.logger.info(f"  测试F1分数: {test_metrics['f1_score']:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            **test_metrics
        }