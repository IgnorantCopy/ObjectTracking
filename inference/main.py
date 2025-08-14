import torch
import os

from utils import config
from data.rd_preprocessor import RDPreprocessor, BatchFile
from data.track_preprocessor import TrajectoryPreprocessor


def main():
    rd_model_config, track_model_config, data_config, _ = config.get_config("./configs/fusion.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root      = data_config['data_root']
    image_seq_len  = data_config['image_seq_len']
    track_seq_len  = data_config['track_seq_len']
    height         = data_config['height']
    width          = data_config['width']
    num_classes    = data_config['num_classes']
    image_channels = data_config['image_channels']
    track_channels = data_config['track_channels']
    config.check_paths(data_root)
    
    # load rd model
    rd_model_path = "./ckpt/rd_models/model_state_dict.pth"
    rd_model = config.get_rd_model(rd_model_config, image_channels, num_classes)
    rd_model.load_state_dict(torch.load(rd_model_path))
    rd_model.to(device)
    print(f"Loaded checkpoint from {rd_model_path}")
    
    # load track model
    track_model_path = "./ckpt/track_models/model_state_dict.pth"
    track_model = config.get_track_model(track_model_config, track_channels, num_classes, track_seq_len)
    track_model.load_state_dict(torch.load(track_model_path))
    track_model.to(device)
    print(f"Loaded checkpoint from {track_model_path}")
    
    # load stacking model
    stacking_model_path = "./ckpt/stacking_models/model_state_dict.pth"
    model = config.get_stacking([rd_model], [track_model], num_classes)
    model.load_state_dict(torch.load(stacking_model_path))
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {stacking_model_path}")
    
    rd_preprocessor = RDPreprocessor((image_channels, height, width), image_seq_len)
    track_preprocessor = TrajectoryPreprocessor(data_root, track_seq_len, num_classes, track_channels, test=False)
    
    # example input
    batch_file = BatchFile(
        1,
        1,
        os.path.join(data_root, "原始回波/1_Label_1.dat"),
        os.path.join(data_root, "点迹/PointTracks_1_1_21.txt"),
        os.path.join(data_root, "航迹/Tracks_1_1_21.txt")
    )
    
    point_index, images, extra_features, missing_rate, image_mask = rd_preprocessor.preprocess(batch_file)
    sequences = track_preprocessor.process_single_trajectory(batch_file)
    if sequences is None:
        return []
    if images is not None:
        images = images.to(device)
        images = images.unsqueeze(0)
    track_features = sequences.T

    pred = []
    last_logits = None
    for t in range(1, track_seq_len + 1):
        if images is not None:
            index_mask_t = (point_index <= t)
            image_mask_t = image_mask * index_mask_t
            image_mask_t = image_mask_t.to(device)

            rd_t = image_mask.float().sum().int()
            if rd_t == 0:
                extra_features_t = torch.tensor([0. for _ in range(extra_features.shape[1])])
            else:
                extra_features_t = extra_features[:rd_t].mean(0)
            extra_features_t = extra_features_t.to(device)
            missing_rate_t = missing_rate[t - 1].float().to(device)

            extra_features_t = extra_features_t.unsqueeze(0)
            missing_rate_t = missing_rate_t.unsqueeze(0)
            image_mask_t = image_mask_t.unsqueeze(0)

        track_features_t = track_features[:, :t]
        track_features_t = track_features_t.to(device)

        track_features_t = track_features_t.unsqueeze(0)
        output_t = model(track_features_t, last_logits, images, extra_features_t, missing_rate_t, image_mask_t)
        last_logits = output_t
        _, pred_t = output_t.max(1)
        pred.append(pred_t.cpu().numpy()[0])
        track_features[t - 1, -1] = pred_t[0]
    
    return pred


if __name__ == '__main__':
    print(main())