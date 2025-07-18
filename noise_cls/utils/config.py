import os
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def check_paths(*paths):
    for path in paths:
        if path is None:
            continue
        if not os.path.exists(path):
            os.makedirs(path)


def get_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config["model"], config["data"], config["train"]


def get_model(config, input_dim, num_classes):
    if config["name"] == "fc":
        from noise_cls.models.fc import FC
        hidden_dim = config["hidden_dim"]
        dropout = config["dropout"]
        model = FC(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes, dropout=dropout)
    else:
        raise NotImplementedError(f"Model {config['name']} not implemented")

    return model


def get_optimizer(config, model, lr):
    if config['name'] == 'Adam':
        weight_decay = config['weight_decay']
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config['name'] == 'SGD':
        weight_decay = config['weight_decay']
        momentum = config['momentum']
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise NotImplementedError(f"Optimizer {config['name']} not implemented")


def get_lr_scheduler(config, optimizer):
    if config['name'] == 'ReduceLROnPlateau':
        factor = config['factor']
        patience = config['patience']
        min_lr = config['min_lr']
        return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
    else:
        raise NotImplementedError(f"LR Scheduler {config['name']} not implemented")


def get_criterion(config):
    if config['name'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss {config['name']} not implemented")