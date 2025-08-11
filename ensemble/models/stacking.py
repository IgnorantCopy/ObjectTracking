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

    def forward(self, track_features, last_logits, images=None, extra_features=None, missing_rate=None, image_mask=None):
        outputs = []
        if images is not None:
            for model in self.rd_models:
                output = model(images, extra_features=extra_features, missing_rate=missing_rate, mask=image_mask)
                outputs.append(output)
        for model in self.track_models:
            output = model(track_features, last_logits)
            outputs.append(output['logits'])
        outputs = torch.cat(outputs, dim=1).to(track_features.device)
        outputs = self.classifier(outputs) if images is not None else outputs
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
        self.need_pred_score = True

    def add_timestep(self, track_features: torch.Tensor, max_len: int,
                     images: torch.Tensor = None, extra_features: torch.Tensor = None,
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
        if images is not None:
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
                        if should_stop and len(temp_counts) > len(counts):
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
            'stop_timestep': self.stop_timestep if self.stopped_early else self.max_seq_len,
            'total_timesteps': self.max_seq_len
        }