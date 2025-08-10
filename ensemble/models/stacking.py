import torch
import torch.nn as nn
from torch.nn.modules.module import T
from typing import Dict
import numpy as np


class Stacking(nn.Module):
    def __init__(self, rd_models: list, track_models: list, num_classes: int):
        super().__init__()
        self.rd_models = rd_models
        self.track_models = track_models
        self.classifier = nn.Sequential(
            nn.Linear((len(rd_models) + len(track_models)) * num_classes, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.freeze()

    def forward(self, track_features, last_logits, images, extra_features=None, missing_rate=None, image_mask=None):
        outputs = []
        for model in self.rd_models:
            output = model(images, extra_features=extra_features, missing_rate=missing_rate, mask=image_mask)
            outputs.append(output)
        for model in self.track_models:
            output = model(track_features, last_logits)
            outputs.append(output['logits'])
        outputs = torch.cat(outputs, dim=1).to(images.device)
        outputs = self.classifier(outputs)
        outputs = self.softmax(outputs)
        return outputs

    def freeze(self):
        for model in self.rd_models:
            for param in model.parameters():
                param.requires_grad = False
        for model in self.track_models:
            for param in model.parameters():
                param.requires_grad = False

    def train(self: T, mode: bool = True) -> T:
        self.training = mode
        self.classifier.train(mode)
        return self

    def eval(self: T) -> T:
        for model in self.rd_models:
            model.eval()
        for model in self.track_models:
            model.eval()
        return self.train(False)


class InferenceEngine:
    """流式推理引擎"""

    def __init__(self, model: Stacking, confidence_threshold: float = 0.95):
        self.model = model
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.max_seq_len = model.track_models[0].max_seq_len

        # 重置状态
        self.reset()

    def reset(self):
        """重置推理状态"""
        self.predictions_history = []
        self.last_logits = None
        self.stopped_early = False
        self.stop_timestep = None

    def add_timestep(self, track_features: torch.Tensor, images: torch.Tensor, extra_features: torch.Tensor = None,
                     missing_rate: torch.Tensor = None, image_mask: torch.Tensor = None) -> Dict[str, any]:
        # 添加到当前序列
        seq_len = track_features.shape[1]
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
        track_features = track_features.unsqueeze(0)
        images = images.unsqueeze(0)
        if extra_features is not None:
            extra_features = extra_features.unsqueeze(0)
        if missing_rate is not None:
            missing_rate = missing_rate.unsqueeze(0)
        if image_mask is not None:
            image_mask = image_mask.unsqueeze(0)

        if torch.cuda.is_available():
            track_features = track_features.cuda()

        # 进行预测
        with torch.no_grad():
            output = self.model(track_features, self.last_logits, images, extra_features=extra_features,
                                missing_rate=missing_rate, image_mask=image_mask)
            self.last_logits = output
            confidence, predictions = torch.max(output, 1)
            should_stop = confidence > self.confidence_threshold

            prediction = predictions[0].item()
            conf = confidence[0].item()

            # 记录历史
            self.predictions_history.append(prediction)

            # 检查是否应该早期停止
            if should_stop:
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
            'stop_timestep': self.stop_timestep if self.stopped_early else self.max_seq_len,
            'total_timesteps': self.max_seq_len
        }