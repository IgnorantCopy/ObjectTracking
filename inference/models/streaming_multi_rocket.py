"""
支持流式推理的MultiRocket模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tsai.models.MultiRocketPlus import MultiRocketBackbonePlus
from typing import Dict, List, Optional, Tuple
import numpy as np

from inference.data.track_preprocessor import TrajectoryPreprocessor

class MultiRocket(nn.Module):
    def __init__(self, c_in: int,
                 c_out: int,
                 seq_len: int,
                 num_features: int = 20_000,
                 dropout: float = 0.2,
                 **kwargs):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.num_features = num_features
        self.dropout = dropout

        self.backbone = MultiRocketBackbonePlus(c_in, seq_len, num_features=num_features, **kwargs)
        backbone_out_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(backbone_out_features),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_features, c_out),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out)
        )


    def forward(self, x: torch.Tensor, last_logits: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.fc(x)
        if last_logits is None:
            last_logits = torch.ones((batch_size, self.c_out), dtype=x.dtype, device=x.device) / self.c_out
        x = x + last_logits
        x = self.head(x)
        return x


class StreamingMultiRocketClassifier(nn.Module):
    """支持流式推理的MultiRocket分类器"""
    
    def __init__(self, 
                 c_in: int, 
                 c_out: int, 
                 max_seq_len: int, 
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
            **kwargs: 传递给backbone的其他参数
        """
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.dropout = dropout
        self.confidence_threshold = confidence_threshold
        
        # 创建多个不同序列长度的backbone
        self.models = nn.ModuleList()

        # 创建不同长度的模型
        self.supported_lengths = self._get_supported_lengths()

        for seq_len in range(1, self.max_seq_len + 1):
            target_len = self._find_best_length(seq_len)
            self.models.append(MultiRocket(
                c_in=c_in,
                c_out=c_out,
                seq_len=target_len,
                num_features=num_features,
                dropout=dropout,
                **kwargs,
            ))

    def _get_supported_lengths(self) -> List[int]:
        """获取支持的序列长度列表"""
        # MultiRocket的最小工作长度是 10
        lengths = []
        current = 10

        # 测试每个长度是否可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        while current <= self.max_seq_len:
            # 测试这个长度是否可用
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

    def forward(self, x: torch.Tensor, last_logits: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 (batch, features, seq_len)
            last_logits: 上一时刻的预测结果 (batch, c_out)
        Returns:
            包含预测结果和置信度的字典
        """
        batch_size, features, seq_len = x.shape
        if seq_len < 10:
            temp = torch.zeros((batch_size, features, 10), dtype=x.dtype, device=x.device)
            temp[:, :, : seq_len] = x
            temp[:, -2, seq_len:] = 1
            temp[:, -1, :] = -1
            x = temp
            del temp
        target_len = self._find_best_length(seq_len)
        # 使用对应长度的模型进行预测
        if x.shape[2] < target_len:
            padding_data = []
            for i in range(batch_size):
                padding_data.append(TrajectoryPreprocessor.data_padding(x[i].cpu().numpy().T, target_len, N=4).T)
            padding_data = torch.from_numpy(np.stack(padding_data, axis=0)).float().to(x.device)
            x = torch.cat([x, padding_data], dim=2)
        logits = self.models[seq_len - 1](x, last_logits)

        # 计算置信度
        probs = F.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]

        return {
            'logits': logits,
            'probabilities': probs,
            'max_probability': max_probs,
        }
    
    def predict_streaming(self, x: torch.Tensor, last_logits: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        流式预测：给定当前序列，返回预测、置信度和是否应该早期停止
        
        Args:
            x: 当前序列 (batch, features, current_seq_len)
            last_logits: 上一时刻的预测结果 (batch, c_out)
        Returns:
            (predictions, confidence, should_stop)
        """
        # 获取预测结果
        with torch.no_grad():
            output = self.forward(x, last_logits)
            logits = output['logits']
            predictions = torch.argmax(logits, dim=1)
            confidence = output['max_probability']
            
            # 判断是否应该早期停止
            should_stop = confidence > self.confidence_threshold
        
        return logits, predictions, confidence, should_stop


class StreamingTrainer:
    """流式模型的训练器"""
    
    def __init__(self, model: StreamingMultiRocketClassifier, device: torch.device):
        self.model = model
        self.device = device
        
    def _compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算准确率"""
        corrects = torch.zeros((predictions.shape[1], self.model.c_out), dtype=torch.float32, device=self.device)
        totals = torch.zeros((predictions.shape[1], self.model.c_out), dtype=torch.float32, device=self.device)
        for i in range(predictions.shape[0]):
            prediction = predictions[i]
            label = labels[i].item()
            for j in range(predictions.shape[1]):
                pred = prediction[j].item()
                corrects[j][label] += float(pred == label)
                totals[j][label] += 1.0

        return corrects, totals

    def train_step(self, 
                   sequences: torch.Tensor, 
                   labels: torch.Tensor, 
                   optimizers: List[torch.optim.Optimizer]) -> Dict[str, float]:
        """
        训练一步
        
        Returns:
            训练指标字典
        """
        batch_size, features, max_len = sequences.shape

        for optimizer in optimizers:
            optimizer.zero_grad()

        total_loss = 0.0
        losses = []
        predictions = []
        is_begin = [False for _ in range(batch_size)]
        begin_time = [max_len for _ in range(batch_size)]
        last_logits = torch.ones((batch_size, self.model.c_out), dtype=sequences.dtype, device=sequences.device) / self.model.c_out
        for length in range(1, self.model.max_seq_len + 1):
            if length > max_len:
                continue
            else:
                x_length = sequences[:, :, :length]

            # 获取预测结果
            output = self.model(x_length, last_logits)
            logits = output['logits']
            last_logits = logits
            # 计算分类损失
            loss = F.cross_entropy(logits, labels)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred)
            # 更新标签列
            sequences[:, -1, length - 1] = pred.float()
            # 判断开始时间
            for i in range(batch_size):
                if is_begin[i]:
                    continue
                if pred[i] == labels[i]:
                    is_begin[i] = True
                    begin_time[i] = length
            total_loss += loss
            losses.append(loss.item())

        total_loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        losses = torch.tensor(losses, device=self.device)
        predictions = torch.stack(predictions, dim=1)   # (batch, timesteps)
        begin_time = sum(begin_time)
        corrects, totals = self._compute_accuracy(predictions, labels)

        return {
            'loss': losses,
            'predictions': predictions,
            'corrects': corrects,
            'totals': totals,
            'begin_time': begin_time
        }

    def evaluate_step(self,
                      sequences: torch.Tensor,
                      labels: torch.Tensor,) -> Dict[str, float]:
        batch_size, features, max_len = sequences.shape

        losses = []
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
            loss = F.cross_entropy(logits, labels)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred)
            # 更新标签列
            sequences[:, -1, length - 1] = pred.float()
            # 判断开始时间
            for i in range(batch_size):
                if is_begin[i]:
                    continue
                if pred[i] == labels[i]:
                    is_begin[i] = True
                    begin_time[i] = length
            losses.append(loss.item())

        losses = torch.tensor(losses, device=self.device)
        predictions = torch.stack(predictions, dim=1)
        begin_time = sum(begin_time)
        corrects, totals = self._compute_accuracy(predictions, labels)

        return {
            'loss': losses,
            'predictions': predictions,
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
        self.predictions_history = []
        self.last_logits = None
        self.stopped_early = False
        self.stop_timestep = None
        self.need_pred_score = True

    def add_timestep(self, features: torch.Tensor, max_len: int) -> Dict[str, any]:
        """
        添加新的时间步

        Args:
            features: (features, seq_len)

        Returns:
            当前预测结果字典
        """
        # 添加到当前序列
        seq_len = features.shape[1]
        if self.stopped_early:
            prediction = self.predictions_history[-1]
            self.predictions_history.append(prediction)
            return {
                'current_length': seq_len,
                'prediction': prediction,
                'should_stop': True,
                'stopped_early': self.stopped_early,
                'stop_timestep': self.stop_timestep,
                'predictions_history': self.predictions_history.copy(),
            }

        # 转换为tensor
        sequence_tensor = features.unsqueeze(0)  # (1, features, timesteps)

        if torch.cuda.is_available():
            sequence_tensor = sequence_tensor.cuda()

        # 进行预测
        with torch.no_grad():
            logits, predictions, confidence, should_stop = self.model.predict_streaming(sequence_tensor,
                                                                                        self.last_logits)
            self.last_logits = logits

            prediction = predictions[0].item()
            conf = confidence[0].item()

            # ==============
            # 进行强制早停判断
            different_threshold = np.floor(max_len * 0.1)
            current_length = len(self.predictions_history)

            # 添加当前预测到临时历史中进行分析
            temp_history = self.predictions_history + [prediction]
            modified_prediction = False

            if current_length >= 10:
                # 情况1：当前长度已经大于等于10
                unique_classes, counts = np.unique(self.predictions_history, return_counts=True)
                max_count = np.max(counts)
                max_classes = unique_classes[counts == max_count]

                # 检查是否满足强制早停条件
                if len(max_classes) == 1 and max_count > different_threshold:
                    # 最多的类别是唯一的，且个数超过阈值，强制早停
                    self.stopped_early = True
                    self.stop_timestep = seq_len
                    final_prediction = max_classes[0]
                    self.predictions_history.append(final_prediction)

                    return {
                        'current_length': seq_len,
                        'prediction': final_prediction,
                        'confidence': conf,
                        'should_stop': True,
                        'stopped_early': self.stopped_early,
                        'stop_timestep': self.stop_timestep,
                        'predictions_history': self.predictions_history.copy(),
                    }
                else:
                    # 不满足强制早停条件，继续正常预测
                    self.predictions_history.append(prediction)

            else:
                # 情况2：当前长度小于10
                if len(self.predictions_history) > 0 and self.need_pred_score:
                    unique_classes, counts = np.unique(self.predictions_history, return_counts=True)
                    max_count = np.max(counts)
                    max_classes = unique_classes[counts == max_count]  # 当前的所有最多类别

                    # 统计加入当前prediction后的分布
                    temp_unique, temp_counts = np.unique(temp_history, return_counts=True)

                    # 计算所有非最多类别的总数
                    non_max_total = 0
                    for i, temp_class in enumerate(temp_unique):
                        # 最多只能有一个类别的最终的预测类别
                        # 因此其他相同数目的非最多类别全部计入总数内
                        if temp_class != max_classes[0]:
                            non_max_total += temp_counts[i]

                    # 检查非最多类别的总数是否超过阈值
                    if non_max_total > different_threshold:
                        # 非最多类别总数超过阈值，可能需要修改预测
                        if conf > 0.8 and len(temp_counts) > len(counts):
                            # 满足早停且是新的预测类别
                            self.need_pred_score = False
                        else:
                            temp_max_count = np.max(temp_counts)
                            temp_max_classes = temp_unique[temp_counts == temp_max_count]
                            # 优先选择在加入新预测后仍然是最多的类别作为替代预测
                            selected_class = None
                            for current_max_class in max_classes:
                                if current_max_class in temp_max_classes:
                                    selected_class = current_max_class
                                    break

                            # 如果当前最多类别在加入新预测后都不是最多的，选择第一个当前最多类别
                            if selected_class is None:
                                selected_class = max_classes[0]

                            prediction = selected_class
                            modified_prediction = True

                    self.predictions_history.append(prediction)

                else:
                    # 第一个预测，直接添加
                    self.predictions_history.append(prediction)

            # 检查是否应该早期停止
            if should_stop and not modified_prediction:
                self.stopped_early = True
                self.stop_timestep = seq_len

            return {
                'current_length': seq_len,
                'prediction': prediction,
                'confidence': conf,
                'should_stop': should_stop,
                'stopped_early': self.stopped_early,
                'stop_timestep': self.stop_timestep,
                'predictions_history': self.predictions_history.copy(),
            }

    def get_final_prediction(self) -> Dict[str, any]:
        """获取最终预测结果"""
        predictions = np.array(self.predictions_history)
        unique, counts = np.unique(predictions, return_counts=True)
        index = np.argmax(counts)
        prediction = unique[index]
        rate = len(predictions[predictions == prediction]) / len(predictions)

        return {
            'prediction': prediction,
            'rate': rate,
            'stop_timestep': self.stop_timestep if self.stopped_early else self.model.max_seq_len,
            'total_timesteps': self.model.max_seq_len
        }