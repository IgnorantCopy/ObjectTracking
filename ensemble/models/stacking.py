import torch
import torch.nn as nn
from torch.nn.modules.module import T


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

    def forward(self, track_features, images, extra_features=None, missing_rate=None, image_mask=None):
        outputs = []
        for model in self.rd_models:
            output = model(images, extra_features=extra_features, missing_rate=missing_rate, mask=image_mask)
            outputs.append(output)
        for model in self.track_models:
            output = model(track_features)
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
        self.fc.train(mode)
        return self

    def eval(self: T) -> T:
        return self.train(False)