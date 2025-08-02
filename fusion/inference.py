import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing

from utils import config, visualize
from fusion.data import dataset


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default='./configs/swin.yaml', help="path to config file")
    parser.add_argument("--device",      type=str, default="cuda",                 help="device to use", choices=["cuda", "cpu"])
    parser.add_argument("--pretrain",    type=str, required=True,                  help="path to the pretrain model")
    args = parser.parse_args()
    print(args)
    return args


def train(model, train_loader, optimizer, criterion, alpha, threshold, device, num_classes, track_seq_len, use_flash_attn):
    model.train()
    train_loss = 0.
    train_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    conf_mat = np.zeros((num_classes, num_classes))
    train_max_begin_time = 0
    train_avg_rate = 0.
    for i, (_, point_index, image, track_features, image_mask, track_mask, label) \
            in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        image = image.to(device)
        track_features = track_features.to(device)
        if use_flash_attn:
            image = image.half()
            track_features = track_features.half()
        else:
            image = image.float()
            track_features = track_features.float()
        label = label.to(device)
        optimizer.zero_grad()

        cls_loss = 0.
        begin_times = [track_seq_len for _ in range(len(image))]
        pred = []
        begin = [False for _ in range(len(image))]
        for t in range(track_seq_len):
            index_mask_t = (point_index <= t + 1)
            image_mask_t = image_mask * index_mask_t
            track_mask_t = track_mask.clone()
            track_mask_t[:, t+1:] = 0

            track_mask_t = track_mask_t.to(device)
            image_mask_t = image_mask_t.to(device)
            output_t = model(track_features, image, track_mask_t, image_mask_t)
            output_max_t, pred_t = output_t.max(1)
            pred_copy = pred_t.clone()
            pred_copy[output_max_t < threshold] = -1

            for j in range(len(pred_copy)):
                if pred_copy[j] != -1 and not begin[j]:
                    begin[j] = True
                    begin_times[j] = t
                elif pred_copy[j] == -1 and begin[j]:
                    pred_copy[j] = pred_t[j]
            pred.append(pred_copy.cpu().tolist())

            cls_loss_t = criterion(output_t, label)
            cls_loss += cls_loss_t
            # 释放显存
            del output_t, output_max_t, pred_t, pred_copy, index_mask_t, image_mask_t, track_mask_t
            torch.cuda.empty_cache()

        cls_loss /= track_seq_len
        begin_times = torch.tensor(begin_times, dtype=torch.float, device=device)
        train_max_begin_time = max(train_max_begin_time, begin_times.max().item())
        time_loss = nn.MSELoss()(begin_times, torch.zeros_like(begin_times))
        time_loss /= track_seq_len ** 2
        loss = alpha * cls_loss + (1 - alpha) * time_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pred = np.array(pred).T   # [batch_size, seq_len]
        label = label.cpu().numpy()
        totals += np.bincount(label, minlength=num_classes)
        for j in range(len(pred)):
            gt = label[j]
            pred_j = pred[j]
            pred_j = pred_j[pred_j != -1]
            if len(pred_j) == 0:
                continue
            unique_vals, counts = np.unique(pred_j, return_counts=True)
            pred_label = unique_vals[counts.argmax()]
            train_avg_rate += len(pred_j[pred_j == pred_label]) / len(pred_j)
            if pred_label == gt:
                corrects[gt] += 1
            conf_mat[pred_label][gt] += 1

    train_acc = corrects.sum() / totals.sum()
    totals[totals == 0] = 1
    train_accuracies = corrects / totals
    train_loss = train_loss / len(train_loader)
    train_avg_rate /= len(train_loader) * track_seq_len
    return train_accuracies, train_loss, train_acc, train_time, train_max_begin_time, train_avg_rate, conf_mat


def val(model, val_loader, criterion, alpha, threshold, device, num_classes, track_seq_len, use_flash_attn):
    model.eval()
    val_loss = 0.
    val_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    conf_mat = np.zeros((num_classes, num_classes))
    val_max_begin_time = 0
    val_avg_rate = 0.
    with torch.no_grad():
        for i, (_, point_index, image, track_features, image_mask, track_mask, label) \
                in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
            image = image.to(device)
            track_features = track_features.to(device)
            if use_flash_attn:
                image = image.half()
                track_features = track_features.half()
            else:
                image = image.float()
                track_features = track_features.float()
            label = label.to(device)

            cls_loss = 0.
            begin_times = [track_seq_len for _ in range(len(image))]
            pred = []
            begin = [False for _ in range(len(image))]
            for t in range(track_seq_len):
                index_mask_t = (point_index <= t + 1)
                image_mask_t = image_mask * index_mask_t
                track_mask_t = track_mask.clone()
                track_mask_t[:, t+1:] = 0

                track_mask_t = track_mask_t.to(device)
                image_mask_t = image_mask_t.to(device)
                output_t = model(track_features, image, track_mask_t, image_mask_t)
                output_max_t, pred_t = output_t.max(1)
                pred_copy = pred_t.clone()
                pred_copy[output_max_t < threshold] = -1

                for j in range(len(pred_copy)):
                    if pred_copy[j] != -1 and not begin[j]:
                        begin[j] = True
                        begin_times[j] = t
                    elif pred_copy[j] == -1 and begin[j]:
                        pred_copy[j] = pred_t[j]
                pred.append(pred_copy.cpu().tolist())

                cls_loss_t = criterion(output_t, label)
                cls_loss += cls_loss_t
                # 释放显存
                del output_t, output_max_t, pred_t, pred_copy, index_mask_t, image_mask_t, track_mask_t
                torch.cuda.empty_cache()
            cls_loss /= track_seq_len
            begin_times = torch.tensor(begin_times)
            val_max_begin_time = max(val_max_begin_time, begin_times.max().item())
            time_loss = nn.MSELoss()(begin_times.float().to(device), torch.zeros_like(begin_times).float().to(device))
            time_loss /= track_seq_len ** 2
            loss = alpha * cls_loss + (1 - alpha) * time_loss
            val_loss += loss.item()

            pred = np.array(pred).T   # [batch_size, seq_len]
            label = label.cpu().numpy()
            totals += np.bincount(label, minlength=num_classes)
            for j in range(len(pred)):
                gt = label[j]
                pred_j = pred[j]
                pred_j = pred_j[pred_j != -1]
                if len(pred_j) == 0:
                    continue
                unique_vals, counts = np.unique(pred_j, return_counts=True)
                pred_label = unique_vals[counts.argmax()]
                val_avg_rate += len(pred_j[pred_j == pred_label]) / len(pred_j)
                if pred_label == gt:
                    corrects[gt] += 1
                conf_mat[pred_label][gt] += 1
    val_acc = corrects.sum() / totals.sum()
    totals[totals == 0] = 1
    val_accuracies = corrects / totals
    val_loss = val_loss / len(val_loader)
    val_avg_rate /= len(val_loader) * track_seq_len

    return val_accuracies, val_loss, val_acc, val_time, val_max_begin_time, val_avg_rate, conf_mat


def main():
    args = config_parser()
    config_path   = args.config_path
    device        = args.device if torch.cuda.is_available() else "cpu"
    pretrain      = args.pretrain

    rd_model_config, track_model_config, fc_model_config, data_config, train_config = config.get_config(config_path)

    use_flash_attn = rd_model_config['name'] in ['Vit', 'ViViT']

    data_root     = data_config['data_root']
    val_ratio     = data_config['val_ratio']
    shuffle       = data_config['shuffle']
    image_seq_len = data_config['image_seq_len']
    track_seq_len = data_config['track_seq_len']
    config.check_paths(data_root)

    batch_size       = train_config['batch_size']
    num_workers      = train_config['num_workers']
    num_classes      = train_config['num_classes']
    channels         = train_config['channels']
    alpha            = train_config['alpha']
    threshold        = train_config['threshold']
    height           = train_config['height']
    width            = train_config['width']
    loss_config      = train_config['loss']

    num_classes -= 2    # exclude noise and unknown class
    model = config.get_model(rd_model_config, track_model_config, channels, num_classes)
    params = torch.load(pretrain, weights_only=False)['state_dict']
    model.load_state_dict(params)

    criterion = config.get_criterion(loss_config)

    model.to(device)

    _, val_transform = config.get_transform(channels, height, width)
    _, val_batch_files = dataset.split_train_val(data_root, num_classes, val_ratio, shuffle)
    val_dataset = dataset.FusedDataset(val_batch_files, image_transform=val_transform, image_seq_len=image_seq_len,
                                       track_seq_len=track_seq_len, track_transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)

    val_accuracies, val_loss, val_acc, val_time, val_max_begin_time, val_avg_rate, val_conf_mat = \
        val(model, val_loader, criterion, alpha, threshold, device, num_classes, track_seq_len, use_flash_attn)

    print(f"Val Loss: {val_loss:.3f}\n"
          f"Val Accuracy: {val_acc:.3f}\n"
          f"Val Time: {time.time() - val_time:.3f}s\n"
          f"Val Max Begin Time: {val_max_begin_time}\n"
          f"Val Avg Rate: {val_avg_rate:.3f}")
    print(f"Confusion Matrix:\n{val_conf_mat}")

    print(f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
