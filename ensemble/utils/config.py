import os
import yaml
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from ensemble.utils import loss


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


def get_model(rd_model_config, channels, num_classes):
    if rd_model_config['name'] == "SwinTransformer":
        from ensemble.models.swin import SwinTransformer3D
        patch_depth        = rd_model_config["patch_depth"]
        patch_height       = rd_model_config["patch_height"]
        patch_width        = rd_model_config["patch_width"]
        embed_dim          = rd_model_config["embed_dim"]
        depths             = rd_model_config["depths"]
        heads              = rd_model_config["heads"]
        window_depth       = rd_model_config["window_depth"]
        window_height      = rd_model_config["window_height"]
        window_width       = rd_model_config["window_width"]
        ff_ratio           = rd_model_config["ff_ratio"]
        qkv_bias           = rd_model_config["qkv_bias"]
        dropout            = rd_model_config["dropout"]
        attn_dropout       = rd_model_config["attn_dropout"]
        dropout_path       = rd_model_config["dropout_path"]
        patch_norm         = rd_model_config["patch_norm"]
        frozen_stages      = rd_model_config["frozen_stages"]
        extra_features_dim = rd_model_config["extra_features_dim"]
        norm          = rd_model_config["norm"]
        if norm == "LayerNorm":
            norm = nn.LayerNorm
        else:
            raise NotImplementedError(f"Norm {norm} not implemented")
        model = SwinTransformer3D(patch_size=(patch_depth, patch_height, patch_width),
                                  num_classes=num_classes, extra_features_dim=extra_features_dim,
                                  in_channels=channels, embed_dim=embed_dim, depths=depths, heads=heads,
                                  window_size=(window_depth, window_height, window_width),
                                  qkv_bias=qkv_bias, dropout=dropout, attn_dropout=attn_dropout,
                                  dropout_path=dropout_path, ff_ratio=ff_ratio, norm=norm,
                                  patch_norm=patch_norm, frozen_stages=frozen_stages)
    else:
        raise NotImplementedError(f"RD Model {rd_model_config['name']} not implemented")

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
        factor   = config['factor']
        patience = config['patience']
        min_lr   = config['min_lr']
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


def get_transform(channels, height, width):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5 for _ in range(channels)],
                             std=[0.5 for _ in range(channels)]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5 for _ in range(channels)],
                             std=[0.5 for _ in range(channels)]),
    ])
    return train_transform, val_transform


def get_scaler(track_model_path, track_seq_len):
    scalers = []
    for i in range(1, track_seq_len + 1):
        scaler_path = os.path.join(track_model_path, f"scaler_{i}.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        scalers.append(scaler)
    return scalers