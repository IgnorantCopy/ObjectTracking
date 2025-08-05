"""
支持流式推理的MultiRocket模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tsai.models.MultiRocketPlus import MultiRocketBackbonePlus
from typing import Dict, List, Optional, Tuple
import numpy as np


class StreamingMultiRocketClassifier(nn.Module):
    """支持流式推理的MultiRocket分类器"""
    
    def __init__(self, 
                 c_in: int, 
                 c_out: int, 
                 max_seq_len: int, 
                 # min_seq_len: int = 10,
                 num_features: int = 20_000,
                 dropout: float = 0.2,
                 confidence_threshold: float = 0.9,
                 **kwargs):
        """
        初始化流式分类器
        
        Args:
            c_in: 输入特征维度
            c_out: 输出类别数
            max_seq_len: 最大序列长度
            min_seq_len: 最小序列长度（开始预测的长度）
            confidence_threshold: 置信度阈值，超过此值可以早期停止
            **kwargs: 传递给backbone的其他参数
        """
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.max_seq_len = max_seq_len
        # self.min_seq_len = min_seq_len
        self.num_features = num_features
        self.dropout = dropout
        self.confidence_threshold = confidence_threshold
        
        # 创建多个不同序列长度的backbone
        self.backbones = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()

        # 创建不同长度的模型
        self.supported_lengths = self._get_supported_lengths()

        for seq_len in self.supported_lengths:
            # 创建backbone
            backbone = MultiRocketBackbonePlus(
                c_in=c_in,
                seq_len=max(10, seq_len),
                num_features=num_features,
                **kwargs
            )
            backbone_out_dim = backbone.num_features
            # 创建对应的分类头
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(backbone_out_dim),
                nn.Dropout(dropout),
                nn.Linear(backbone_out_dim, c_out),
            )
            
            self.backbones[str(seq_len)] = backbone
            self.classifiers[str(seq_len)] = classifier

    def _get_supported_lengths(self) -> List[int]:
        """获取支持的序列长度列表"""
        # 基于测试结果，HydraMultiRocket的最小工作长度是10
        # 但某些长度（如25）可能有问题，需要测试验证
        lengths = []
        current = 1

        # 测试每个长度是否可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        while current <= self.max_seq_len:
            # 测试这个长度是否可用
            if current < 10:
                lengths.append(current)
                current += 1
                continue
            try:
                test_model = MultiRocketBackbonePlus(
                    c_in=self.c_in,
                    seq_len=current,
                    num_features=self.num_features,
                )
                test_model.to(device)
                test_input = torch.randn(1, self.c_in, current, device=device)
                _ = test_model(test_input)
                lengths.append(current)
                del test_model, test_input  # 清理内存
            except Exception as e:
                print(f"序列长度 {current} 不可用: {e}")
            current += 1

        # 确保包含最大长度
        while lengths[-1] < self.max_seq_len:
            current += 1
            try:
                test_model = MultiRocketBackbonePlus(
                    c_in=self.c_in,
                    seq_len=current,
                    num_features=self.num_features,
                )
                test_model.to(device)
                test_input = torch.randn(1, self.c_in, self.max_seq_len, device=device)
                _ = test_model(test_input)
                lengths.append(current)
                del test_model, test_input
            except Exception as e:
                print(f"序列长度 {current} 不可用: {e}")

        return lengths

    def _find_best_length(self, target_len: int) -> int:
        """找到最适合的模型长度"""
        # 找到大于等于target_len的最小支持长度
        for length in self.supported_lengths:
            if length >= target_len:
                return length
        # 如果没找到，返回最大长度
        return self.supported_lengths[-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch, features, seq_len)
        Returns:
            包含预测结果和置信度的字典
        """
        batch_size, features, seq_len = x.shape
        if seq_len < 10:
            temp = torch.zeros((batch_size, features, 10), dtype=x.dtype, device=x.device)
            temp[:, :, : seq_len] = x
            x = temp
            del temp
        target_len = self._find_best_length(seq_len)
        # 使用对应长度的模型进行预测
        if x.shape[2] < target_len:
            # 使用最后一个时间步进行padding
            padding = x[:, :, -1:].repeat(1, 1, target_len - x.shape[2])
            x = torch.cat([x, padding], dim=2)
        features = self.backbones[str(target_len)](x)
        logits = self.classifiers[str(target_len)](features)
        
        # 计算置信度
        probs = F.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]

        return {
            'logits': logits,
            'probabilities': probs,
            'max_probability': max_probs,
        }
    
    def predict_streaming(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        流式预测：给定当前序列，返回预测、置信度和是否应该早期停止
        
        Args:
            x: 当前序列 (batch, features, current_seq_len)
            
        Returns:
            (predictions, confidence, should_stop)
        """
        current_len = x.shape[2]
        
        # # 如果序列太短，返回低置信度
        # if current_len < 10:  # HydraMultiRocket的最小要求是10
        #     batch_size = x.shape[0]
        #     predictions = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        #     confidence = torch.zeros(batch_size, device=x.device)
        #     return predictions, confidence, False
        
        # 获取预测结果
        with torch.no_grad():
            output = self.forward(x, current_len)
            predictions = torch.argmax(output['logits'], dim=1)
            confidence = output['max_probability']
            
            # 判断是否应该早期停止
            should_stop = confidence > self.confidence_threshold
        
        return predictions, confidence, should_stop


class StreamingTrainer:
    """流式模型的训练器"""
    
    def __init__(self, model: StreamingMultiRocketClassifier, device: torch.device):
        self.model = model
        self.device = device
        
    def compute_progressive_loss(self, 
                                sequences: torch.Tensor, 
                                labels: torch.Tensor,
                                length_weights: Optional[Dict[int, float]] = None)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        计算渐进式损失：在不同序列长度下都计算损失
        
        Args:
            sequences: 输入序列 (batch, features, seq_len)
            labels: 标签
            length_weights: 不同长度的权重
            
        Returns:
            总损失
        """
        if length_weights is None:
            # 默认权重：长序列权重更高
            length_weights = {length: np.log(length + 1) for length in range(1, self.model.max_seq_len + 1)}
            # 归一化权重
            total_weight = sum(length_weights.values())
            length_weights = {k: v / total_weight for k, v in length_weights.items()}
        
        total_loss = 0.0
        batch_size, features, max_len = sequences.shape

        predictions = []
        is_begin = [False for _ in range(batch_size)]
        begin_time = [max_len for _ in range(batch_size)]
        for length in range(1, self.model.max_seq_len + 1):
            if length > max_len:
                continue
            else:
                x_length = sequences[:, :, :length]

            # 获取预测结果
            output = self.model(x_length)
            logits = output['logits']
            # 计算分类损失
            classification_loss = F.cross_entropy(logits, labels)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred)
            # 判断开始时间
            for i in range(batch_size):
                if is_begin[i]:
                    continue
                if pred[i] == labels[i]:
                    is_begin[i] = True
                    begin_time[i] = length

            # 加权
            weight = length_weights.get(length, 1.0)
            total_loss += weight * classification_loss

        # 计算准确率
        predictions = torch.stack(predictions, dim=1)
        corrects = torch.tensor([0. for _ in range(self.model.c_out)], device=self.device)
        totals = torch.tensor([0. for _ in range(self.model.c_out)], device=self.device)
        for i in range(batch_size):
            prediction = predictions[i]
            counts = torch.bincount(prediction, minlength=self.model.c_out)
            pred = counts.argmax().item()
            label = labels[i].item()
            corrects[label] += float(pred == label)
            totals[label] += 1.0
        return total_loss, corrects, totals, sum(begin_time)
    
    def train_step(self, 
                   sequences: torch.Tensor, 
                   labels: torch.Tensor, 
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        训练一步
        
        Returns:
            训练指标字典
        """
        optimizer.zero_grad()
        
        # 计算渐进式损失
        loss, corrects, totals, begin_time = self.compute_progressive_loss(sequences, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'corrects': corrects,
            'totals': totals,
            'begin_time': begin_time
        }

    def evaluate_step(self,
                 sequences: torch.Tensor,
                 labels: torch.Tensor) -> Dict[str, float]:
        loss, corrects, totals, begin_time = self.compute_progressive_loss(sequences, labels)
        return {
            'loss': loss.item(),
            'corrects': corrects,
            'totals': totals,
            'begin_time': begin_time
        }


class StreamingInferenceEngine:
    """流式推理引擎"""
    
    def __init__(self, model: StreamingMultiRocketClassifier):
        self.model = model
        self.model.eval()
        
        # 重置状态
        self.reset()
    
    def reset(self):
        """重置推理状态"""
        self.current_sequence = []
        self.predictions_history = []
        self.confidence_history = []
        self.final_prediction = None
        self.stopped_early = False
        self.stop_timestep = None
    
    def add_timestep(self, features: np.ndarray) -> Dict[str, any]:
        """
        添加新的时间步
        
        Args:
            features: 新时间步的特征 (feature_dim,)
            
        Returns:
            当前预测结果字典
        """
        # 添加到当前序列
        self.current_sequence.append(features)
        current_length = len(self.current_sequence)
        
        # 转换为tensor
        sequence_array = np.array(self.current_sequence).T  # (features, timesteps)
        sequence_tensor = torch.from_numpy(sequence_array).float().unsqueeze(0)  # (1, features, timesteps)
        
        if torch.cuda.is_available():
            sequence_tensor = sequence_tensor.cuda()
        
        # 进行预测
        with torch.no_grad():
            predictions, confidence, should_stop = self.model.predict_streaming(sequence_tensor)
            
            prediction = predictions[0].item()
            conf = confidence[0].item()
            
            # 记录历史
            self.predictions_history.append(prediction)
            self.confidence_history.append(conf)
            
            # 检查是否应该早期停止
            if should_stop and not self.stopped_early and current_length >= self.model.min_seq_len:
                self.stopped_early = True
                self.stop_timestep = current_length
                self.final_prediction = prediction
            
            result = {
                'current_length': current_length,
                'prediction': prediction,
                'confidence': conf,
                'should_stop': should_stop,
                'stopped_early': self.stopped_early,
                'stop_timestep': self.stop_timestep,
                'predictions_history': self.predictions_history.copy(),
                'confidence_history': self.confidence_history.copy()
            }
            
            return result
    
    def get_final_prediction(self) -> Dict[str, any]:
        """获取最终预测结果"""
        if self.stopped_early:
            return {
                'prediction': self.final_prediction,
                'confidence': self.confidence_history[self.stop_timestep - 1],
                'stopped_at_timestep': self.stop_timestep,
                'total_timesteps': len(self.current_sequence)
            }
        else:
            # 使用最后的预测
            return {
                'prediction': self.predictions_history[-1] if self.predictions_history else None,
                'confidence': self.confidence_history[-1] if self.confidence_history else 0.0,
                'stopped_at_timestep': len(self.current_sequence),
                'total_timesteps': len(self.current_sequence)
            }