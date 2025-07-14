import torch
import torch.nn as nn
from thop import profile


class FusedModel(nn.Module):
    def __init__(self, rd_model, track_model, num_classes):
        super().__init__()
        self.swin_transformer = rd_model
        self.roformer = track_model
        self.head = nn.Linear(self.swin_transformer.num_features + self.roformer.d_model, num_classes)

    def forward(self, track_features, images, track_mask=None, image_mask=None):
        track_features = self.roformer(track_features, track_mask)
        image_features = self.swin_transformer(images, image_mask)
        features = torch.cat([track_features, image_features], dim=1)
        return self.head(features)


if __name__ == '__main__':
    config_path = "../../configs/fusion/fused.yaml"
    from fusion.utils import config
    rd_model_config, track_model_config, data_config, train_config = config.get_config(config_path)
    height = train_config['height']
    width = train_config['width']
    image_seq_len = data_config['image_seq_len']
    track_seq_len = data_config['track_seq_len']
    batch_size = train_config['batch_size']
    input_dim = track_model_config['input_dim']
    model = config.get_model(rd_model_config, track_model_config, channels=1, num_classes=4)
    image = torch.randn(batch_size, image_seq_len, 1, height, width)
    image_mask = torch.ones((batch_size, image_seq_len))
    track_features = torch.randn(batch_size, track_seq_len, input_dim)
    track_mask = torch.ones((batch_size, track_seq_len))
    flops, params = profile(model, inputs=(track_features, image, track_mask, image_mask))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")