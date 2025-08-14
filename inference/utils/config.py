import os
import yaml
import torch.nn as nn


def check_paths(*paths):
    for path in paths:
        if path is None:
            continue
        if not os.path.exists(path):
            os.makedirs(path)


def get_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config["rd_model"], config["track_model"], config["data"], config["train"]


def get_rd_model(rd_model_config, channels, num_classes):
    if rd_model_config['name'] == "SwinTransformer":
        from inference.models.swin import SwinTransformer3D

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


def get_track_model(track_model_config, channels, num_classes, seq_len):
    if track_model_config['name'] == "MultiRocket":
        from inference.models.streaming_multi_rocket import StreamingMultiRocketClassifier

        num_features         = track_model_config["num_features"]
        dropout              = track_model_config["dropout"]
        confidence_threshold = track_model_config["confidence_threshold"]

        model = StreamingMultiRocketClassifier(
            c_in=channels,
            c_out=num_classes,
            max_seq_len=seq_len,
            num_features=num_features,
            dropout=dropout,
            confidence_threshold=confidence_threshold,
        )
    else:
        raise NotImplementedError(f"Track Model {track_model_config['name']} not implemented")

    return model


def get_stacking(rd_models, track_models, num_classes):
    from inference.models.stacking import Stacking
    
    return Stacking(rd_models, track_models, num_classes)