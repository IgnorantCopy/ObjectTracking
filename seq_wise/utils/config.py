import os
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from seq_wise.utils import loss


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


def get_model(config, channels, num_classes, height, width, seq_len):
    if config["name"] == "Vit":
        from seq_wise.models.vit import OuterVit
        dim = config["dim"]
        depth = config["depth"]
        heads = config["heads"]
        mlp_dim = config["mlp_dim"]
        dim_head = config["dim_head"]
        dropout = config["dropout"]
        emb_dropout = config["emb_dropout"]
        patch_height = config["patch_height"]
        patch_width = config["patch_width"]
        return OuterVit(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, image_size=(height, width),
                        patch_size=(patch_height, patch_width), in_channels=channels, num_classes=num_classes,
                        seq_len=seq_len, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
    elif config["name"] == "ViViT":
        from seq_wise.models.vivit import ViViT
        dim = config["dim"]
        depth = config["depth"]
        heads = config["heads"]
        mlp_dim = config["mlp_dim"]
        dim_head = config["dim_head"]
        dropout = config["dropout"]
        emb_dropout = config["emb_dropout"]
        patch_height = config["patch_height"]
        patch_width = config["patch_width"]
        return ViViT(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, image_size=(height, width),
                     patch_size=(patch_height, patch_width), in_channels=channels, num_classes=num_classes,
                     seq_len=seq_len, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
    else:
        raise NotImplementedError(f"Model {config['name']} not implemented")


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
    elif config['name'] == "FocalLoss":
        gamma = config['gamma']
        return loss.FocalLoss(gamma=gamma)
    else:
        raise NotImplementedError(f"Loss {config['name']} not implemented")
