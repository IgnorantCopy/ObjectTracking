import os
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import loss


def check_paths(*paths):
    for path in paths:
        if path is None:
            continue
        if not os.path.exists(path):
            if os.path.isdir(path):
                os.makedirs(path)
            else:
                raise ValueError(f"File {path} does not exist")


def get_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config["model"], config["data"], config["train"]


def get_optimizer(config, model, lr):
    if config['name'] == 'Adam':
        weight_decay = config['weight_decay']
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config['name'] == 'SGD':
        weight_decay = config['weight_decay']
        momentum = config['momentum']
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise NotImplementedError(f"Optimizer {config['name']} not implemented")
    return optimizer


def get_lr_scheduler(config, optimizer):
    if config['name'] == 'ReduceLROnPlateau':
        factor = config['factor']
        patience = config['patience']
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    else:
        raise NotImplementedError(f"LR Scheduler {config['name']} not implemented")
    return lr_scheduler


def get_criterion(config):
    if config['name'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif config['name'] == "FocalLoss":
        gamma = config['gamma']
        criterion = loss.FocalLoss(gamma=gamma)
    else:
        raise NotImplementedError(f"Loss {config['name']} not implemented")
    return criterion