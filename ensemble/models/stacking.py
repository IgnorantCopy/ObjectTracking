import torch
import torch.nn as nn
from torch.nn.modules.module import T


class Stacking(nn.Module):
    def __init__(self, models: list, num_classes: int):
        super().__init__()
        self.models = models
        self.fc = nn.Linear(len(models) * num_classes, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.freeze()

    def forward(self, track_features, time, images, extra_features=None, missing_rate=None, image_mask=None):
        outputs = []
        for model in self.models:
            if isinstance(model, nn.Module):
                output = model(images, extra_features=extra_features, missing_rate=missing_rate, mask=image_mask)
            else:
                output = model.predict_proba(track_features, time)
                output = torch.from_numpy(output).float().to(images.device)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1).to(images.device)
        outputs = self.fc(outputs)
        outputs = self.softmax(outputs)
        return outputs

    def freeze(self):
        for model in self.models:
            if isinstance(model, nn.Module):
                for param in model.parameters():
                    param.requires_grad = False

    def train(self: T, mode: bool = True) -> T:
        self.training = mode
        self.fc.train(mode)
        return self

    def eval(self: T) -> T:
        return self.train(False)