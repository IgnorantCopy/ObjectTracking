import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import multiprocessing

from utils import config
from data import dataset
from utils.logger import Logger
from models.stacking import Stacking, InferenceEngine
from rd.data.dataset import split_train_val


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default='./configs/fusion.yaml', help="path to config file")
    parser.add_argument("--log-path",    type=str, default="./logs",                help="path to log file")
    parser.add_argument("--device",      type=str, default="cuda",                  help="device to use", choices=["cuda", "cpu"])
    parser.add_argument("--rd-model",    type=str, required=True,                   help="path to rd model")
    parser.add_argument("--track-model", type=str, required=True,                   help="path to track model")
    parser.add_argument("--stacking",    type=str, required=True,                   help="path to stacking model")
    parser.add_argument("--result-path", type=str, required=True,                   help="path to store the result file")
    args = parser.parse_args()
    print(args)
    return args


def test(model, data_loader, device, track_seq_len, result_path, use_flash_attn):
    os.makedirs(result_path, exist_ok=True)
    model.eval()

    engine = InferenceEngine(model, confidence_threshold=0.95)
    corrects = 0
    corrects_strict = 0
    totals = 0
    avg_streaming_time = 0.
    avg_begin_time = 0.
    avg_stop_time = 0.
    avg_rate = 0.
    with torch.no_grad():
        for i, (batch_files, point_index, image, track_features, extra_features, missing_rate, image_mask, label) \
                in tqdm(enumerate(data_loader), total=len(data_loader), desc="Test"):
            image = image.to(device)
            if use_flash_attn:
                image = image.half()
            else:
                image = image.float()
            label = label.to(device)

            for batch in range(len(image)):
                engine.reset()
                point_index_batch = point_index[batch]
                image_batch = image[batch]
                track_features_batch = track_features[batch]
                extra_features_batch = extra_features[batch]
                missing_rate_batch = missing_rate[batch]
                image_mask_batch = image_mask[batch]
                label_batch = label[batch]
                batch_file = batch_files[batch]

                start_time = time.time()
                is_begin = False
                timestep_prediction = []
                for t in range(1, track_seq_len + 1):
                    index_mask_t = (point_index_batch <= t)
                    image_mask_t = image_mask_batch * index_mask_t
                    track_features_t = track_features_batch[:t, :].T

                    image_mask_t = image_mask_t.to(device)
                    track_features_t = track_features_t.to(device)

                    rd_t = image_mask_t.float().sum().int()
                    if rd_t == 0:
                        extra_features_t = torch.tensor([0. for _ in range(extra_features_batch.shape[1])])
                    else:
                        extra_features_t = extra_features_batch[:rd_t].mean(0)
                    extra_features_t = extra_features_t.float().to(device)
                    missing_rate_t = missing_rate_batch[t - 1].float().to(device)

                    result = engine.add_timestep(track_features_t, image_batch, extra_features_t, missing_rate_t, image_mask_t)
                    prediction = result['prediction']
                    track_features_batch[t - 1, -1] = prediction
                    timestep_prediction.append(prediction)

                    if not is_begin and prediction == label_batch:
                        is_begin = True
                        avg_begin_time += t
                if not is_begin:
                    avg_begin_time += track_seq_len
                avg_streaming_time += time.time() - start_time

                result = engine.get_final_prediction()
                final_prediction = result['prediction']
                stop_timestep = result['stop_timestep']
                rate = result['rate']

                avg_stop_time += stop_timestep
                avg_rate += rate
                totals += 1
                if final_prediction == label_batch:
                    if rate > 0.9:
                        corrects_strict += 1
                    corrects += 1

                # save result
                track_file = batch_file.track_file
                df = pd.read_csv(track_file, encoding='gbk', header=0)
                num_points = int(os.path.basename(track_file).split("_")[-1].split(".")[0])

                if num_points <= track_seq_len:
                    timestep_prediction = timestep_prediction[:num_points]
                else:
                    timestep_prediction += [timestep_prediction[-1] for _ in range(num_points - track_seq_len)]
                df['识别结果'] = np.array(timestep_prediction) + 1
                df.to_csv(os.path.join(result_path, os.path.basename(track_file)), index=False, encoding='gbk')


    acc = corrects / totals
    acc_strict = corrects_strict / totals
    avg_streaming_time /= totals
    avg_streaming_time *= 1000   # ms
    avg_begin_time /= totals
    avg_stop_time /= totals
    avg_rate /= totals

    return acc, acc_strict, avg_streaming_time, avg_begin_time, avg_stop_time, avg_rate


def main():
    args = config_parser()
    config_path      = args.config_path
    log_path         = args.log_path
    rd_model_path    = args.rd_model
    track_model_path = args.track_model
    stacking         = args.stacking
    device           = args.device if torch.cuda.is_available() else "cpu"
    result_path      = args.result_path

    log_path = os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    config.check_paths(log_path)
    logger = Logger(os.path.join(log_path, "train.txt"))
    rd_model_config, track_model_config, data_config, train_config = config.get_stacking_config(config_path)

    use_flash_attn = rd_model_config['name'] in ['Vit', 'ViViT']

    data_root      = data_config['data_root']
    val_ratio      = data_config['val_ratio']
    shuffle        = data_config['shuffle']
    image_seq_len  = data_config['image_seq_len']
    track_seq_len  = data_config['track_seq_len']
    height         = data_config['height']
    width          = data_config['width']
    num_classes    = data_config['num_classes']
    image_channels = data_config['image_channels']
    track_channels = data_config['track_channels']
    config.check_paths(data_root)

    batch_size       = train_config['batch_size']
    num_workers      = train_config['num_workers']


    rd_model = config.get_rd_model(rd_model_config, image_channels, num_classes)
    if rd_model_path:
        checkpoint = torch.load(rd_model_path, weights_only=False)
        rd_model.load_state_dict(checkpoint['state_dict'])
        logger.log(f"Loaded checkpoint from {rd_model_path}")
    rd_model.to(device)

    track_model = config.get_track_model(track_model_config, track_channels, num_classes, track_seq_len)
    if track_model_path:
        checkpoint = torch.load(track_model_path, weights_only=False)
        track_model.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"Loaded checkpoint from {track_model_path}")
    track_model.to(device)

    model = Stacking([rd_model], [track_model], num_classes)

    if stacking:
        checkpoint = torch.load(stacking, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        logger.log(f"Loaded checkpoint from {stacking}")
    model.to(device)

    _, val_transform = config.get_transform(image_channels, height, width)

    _, val_batch_files = split_train_val(data_root, num_classes, val_ratio, shuffle, False)
    val_dataset = dataset.FusedDataset(val_batch_files, image_transform=val_transform, image_seq_len=image_seq_len,
                                       track_seq_len=track_seq_len, test=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=dataset.FusedDataset.collate_fn)

    val_acc, val_acc_strict, val_avg_streaming_time, val_avg_begin_time, val_avg_stop_time, val_avg_rate = \
        test(model, val_loader, device, track_seq_len, os.path.join(result_path, "航迹"), use_flash_attn)
    logger.log(f"Val Accuracy: {val_acc:.3f}\n"
               f"Val Accuracy Strict: {val_acc_strict:.3f}\n"
               f"Val Avg Streaming Time: {val_avg_streaming_time:.3f}ms\n"
               f"Val Avg Begin Time: {val_avg_begin_time:.3f}\n"
               f"Val Avg Stop Time: {val_avg_stop_time:.3f}\n"
               f"Val Avg Rate: {val_avg_rate:.3f}\n"
               f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    logger.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
