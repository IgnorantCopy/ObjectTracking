import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

from utils import config
from data import dataset
from utils.logger import Logger
from models.stacking import Stacking
from rd.data.dataset import split_train_val


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default='./configs/fusion.yaml', help="path to config file")
    parser.add_argument("--log-path",    type=str, default="./logs",                help="path to log file")
    parser.add_argument("--rd-model",    type=str, required=True,                   help="path to rd model")
    parser.add_argument("--track-model", type=str, required=True,                   help="path to track model")
    parser.add_argument("--resume",      type=str, default=None,                    help="path to checkpoint file")
    parser.add_argument("--device",      type=str, default="cuda",                  help="device to use", choices=["cuda", "cpu"])
    parser.add_argument("--result-path", type=str, default=None,                    help="path to store the result file")
    args = parser.parse_args()
    print(args)
    return args


def train(model, train_loader, optimizer, criterion, device, num_classes, track_seq_len, use_flash_attn):
    model.train()
    train_loss = 0.
    train_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    conf_mat = np.zeros((num_classes, num_classes))
    train_avg_begin_time = 0
    train_avg_rate = 0.
    for i, (_, point_index, image, track_features, extra_features, missing_rate, image_mask, label) \
            in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        image = image.to(device)
        if use_flash_attn:
            image = image.half()
        else:
            image = image.float()
        label = label.to(device)
        optimizer.zero_grad()

        loss = 0.
        begin_times = [track_seq_len for _ in range(len(image))]
        pred = []
        begin = [False for _ in range(len(image))]
        last_logits = torch.ones((len(image), num_classes), dtype=torch.float32, device=device) / num_classes
        for t in range(1, track_seq_len + 1):
            index_mask_t = (point_index <= t)
            image_mask_t = image_mask * index_mask_t
            track_features_t = track_features[:, :t, :]
            track_features_t = track_features_t.transpose(1, 2)

            image_mask_t = image_mask_t.to(device)
            track_features_t = track_features_t.to(device)

            extra_features_t = []
            for j, mask in enumerate(image_mask_t):
                rd_t = mask.float().sum().int()
                if rd_t == 0:
                    extra_features_t.append(torch.tensor([0. for _ in range(extra_features.shape[2])]))
                else:
                    extra_features_t.append(extra_features[j][:rd_t].mean(0))
            extra_features_t = torch.stack(extra_features_t).float().to(device)
            missing_rate_t = missing_rate[:, t - 1].float().to(device)

            output_t = model(track_features_t, last_logits, image, extra_features_t, missing_rate_t, image_mask_t)
            last_logits = output_t
            _, pred_t = output_t.max(1)

            for j in range(len(pred_t)):
                if not begin[j] and pred_t[j] == label[j]:
                    begin[j] = True
                    begin_times[j] = t
            track_features[:, t - 1, -1] = pred_t.float()
            pred.append(pred_t.cpu().tolist())

            cls_loss_t = criterion(output_t, label)
            loss += cls_loss_t

        loss /= track_seq_len
        begin_times = torch.tensor(begin_times, dtype=torch.float, device=device)
        train_avg_begin_time += begin_times.sum().item()
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
            rate = len(pred_j[pred_j == pred_label]) / len(pred_j)
            train_avg_rate += rate
            if pred_label == gt and rate > 0.9:
                corrects[gt] += 1
            conf_mat[pred_label][gt] += 1

    train_acc = corrects.sum() / totals.sum()
    train_avg_begin_time /= totals.sum()
    train_avg_rate /= totals.sum()
    totals[totals == 0] = 1
    train_accuracies = corrects / totals
    train_loss = train_loss / len(train_loader)
    return train_accuracies, train_loss, train_acc, train_time, train_avg_begin_time, train_avg_rate, conf_mat


def val(model, val_loader, criterion, device, num_classes, track_seq_len, use_flash_attn):
    model.eval()
    val_loss = 0.
    val_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    conf_mat = np.zeros((num_classes, num_classes))
    val_avg_begin_time = 0
    val_avg_rate = 0.
    with torch.no_grad():
        for i, (_, point_index, image, track_features, extra_features, missing_rate, image_mask, label) \
                in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
            image = image.to(device)
            if use_flash_attn:
                image = image.half()
            else:
                image = image.float()
            label = label.to(device)

            loss = 0.
            begin_times = [track_seq_len for _ in range(len(image))]
            pred = []
            begin = [False for _ in range(len(image))]
            last_logits = torch.ones((len(image), num_classes), dtype=torch.float32, device=device) / num_classes
            for t in range(1, track_seq_len + 1):
                index_mask_t = (point_index <= t)
                image_mask_t = image_mask * index_mask_t
                track_features_t = track_features[:, :t, :]
                track_features_t = track_features_t.transpose(1, 2)

                image_mask_t = image_mask_t.to(device)
                track_features_t = track_features_t.to(device)

                extra_features_t = []
                for j, mask in enumerate(image_mask_t):
                    rd_t = mask.float().sum().int()
                    if rd_t == 0:
                        extra_features_t.append(torch.tensor([0. for _ in range(extra_features.shape[2])]))
                    else:
                        extra_features_t.append(extra_features[j][:rd_t].mean(0))
                extra_features_t = torch.stack(extra_features_t).float().to(device)
                missing_rate_t = missing_rate[:, t - 1].float().to(device)

                output_t = model(track_features_t, last_logits, image, extra_features_t, missing_rate_t, image_mask_t)
                last_logits = output_t
                output_max_t, pred_t = output_t.max(1)

                for j in range(len(pred_t)):
                    if not begin[j] and pred_t[j] == label[j]:
                        begin[j] = True
                        begin_times[j] = t
                track_features[:, t - 1, -1] = pred_t.float()
                pred.append(pred_t.cpu().tolist())

                cls_loss_t = criterion(output_t, label)
                loss += cls_loss_t

            loss /= track_seq_len
            begin_times = torch.tensor(begin_times)
            val_avg_begin_time += begin_times.sum().item()
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
                rate = len(pred_j[pred_j == pred_label]) / len(pred_j)
                val_avg_rate += rate
                if pred_label == gt and rate > 0.9:
                    corrects[gt] += 1
                conf_mat[pred_label][gt] += 1
    val_acc = corrects.sum() / totals.sum()
    val_avg_rate /= totals.sum()
    val_avg_begin_time /= totals.sum()
    totals[totals == 0] = 1
    val_accuracies = corrects / totals
    val_loss = val_loss / len(val_loader)

    return val_accuracies, val_loss, val_acc, val_time, val_avg_begin_time, val_avg_rate, conf_mat


def test(model, data_loader, device, num_classes, track_seq_len, result_path, use_flash_attn):
    os.makedirs(result_path, exist_ok=True)

    model.eval()
    corrects = 0
    totals = 0
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

            pred = []
            begin = [False for _ in range(len(image))]
            last_logits = torch.ones((len(image), num_classes), dtype=torch.float32, device=device) / num_classes
            for t in range(1, track_seq_len + 1):
                index_mask_t = (point_index <= t)
                image_mask_t = image_mask * index_mask_t
                track_features_t = track_features[:, :t, :]
                track_features_t = track_features_t.transpose(1, 2)

                image_mask_t = image_mask_t.to(device)
                track_features_t = track_features_t.to(device)

                extra_features_t = []
                for j, mask in enumerate(image_mask_t):
                    rd_t = mask.float().sum().int()
                    if rd_t == 0:
                        extra_features_t.append(torch.tensor([0. for _ in range(extra_features.shape[2])]))
                    else:
                        extra_features_t.append(extra_features[j][:rd_t].mean(0))
                extra_features_t = torch.stack(extra_features_t).float().to(device)
                missing_rate_t = missing_rate[:, t - 1].float().to(device)

                output_t = model(track_features_t, last_logits, image, extra_features_t, missing_rate_t, image_mask_t)
                last_logits = output_t
                _, pred_t = output_t.max(1)

                for j in range(len(pred_t)):
                    if not begin[j] and pred_t[j] == label[j]:
                        begin[j] = True
                track_features[:, t - 1, -1] = pred_t.float()
                pred.append(pred_t.cpu().tolist())

                del output_t, pred_t, index_mask_t, image_mask_t
                torch.cuda.empty_cache()

            pred = np.array(pred).T   # [batch_size, seq_len]
            for batch in range(len(batch_files)):
                batch_file = batch_files[batch]

                cls = batch_file.label
                timestep_prediction = pred[batch]
                batch_pred = timestep_prediction[timestep_prediction != -1]
                if len(batch_pred) == 0:
                    pred_label = -1
                else:
                    unique_vals, counts = np.unique(batch_pred, return_counts=True)
                    pred_label = unique_vals[counts.argmax()]
                rate = len(batch_pred[batch_pred == pred_label]) / len(batch_pred)
                avg_rate += rate
                if pred_label == cls - 1 and rate > 0.9:
                    corrects += 1
                totals += 1

                # save result
                track_file = batch_file.track_file
                df = pd.read_csv(track_file, encoding='gbk', header=0)
                num_points = int(os.path.basename(track_file).split("_")[-1].split(".")[0])

                if num_points <= track_seq_len:
                    timestep_prediction = timestep_prediction[:num_points]
                else:
                    timestep_prediction = np.concatenate([
                        timestep_prediction,
                        np.ones(num_points - track_seq_len, dtype=timestep_prediction.dtype) * timestep_prediction[-1]
                    ])
                df['识别结果'] = timestep_prediction + 1
                df.to_csv(os.path.join(result_path, os.path.basename(track_file)), index=False, encoding='gbk')

    acc = corrects / totals
    avg_rate /= totals

    return acc, avg_rate


def save_model(model, optimizer, lr_scheduler, epoch, best_acc, filename):
    torch.save({
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, filename)


def main():
    args = config_parser()
    config_path      = args.config_path
    log_path         = args.log_path
    rd_model_path    = args.rd_model
    track_model_path = args.track_model
    resume           = args.resume
    device           = args.device if torch.cuda.is_available() else "cpu"
    result_path      = args.result_path

    log_path = os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    config.check_paths(log_path)
    writer = SummaryWriter(log_dir=log_path)
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
    epochs           = train_config['num_epochs']
    lr               = train_config['lr']
    lr_config        = train_config['lr_scheduler']
    optimizer_config = train_config['optimizer']
    loss_config      = train_config['loss']


    rd_model = config.get_rd_model(rd_model_config, image_channels, num_classes)
    if rd_model_path:
        checkpoint = torch.load(rd_model_path, weights_only=False)
        rd_model.load_state_dict(checkpoint['state_dict'])
        logger.log(f"Loaded checkpoint from {rd_model_path}")
    if use_flash_attn:
        rd_model.half()
    rd_model.to(device)

    track_model = config.get_track_model(track_model_config, track_channels, num_classes, track_seq_len)
    if track_model_path:
        checkpoint = torch.load(track_model_path, weights_only=False)
        track_model.load_state_dict(checkpoint['model_state_dict'])
        logger.log(f"Loaded checkpoint from {track_model_path}")
    track_model.to(device)

    model = Stacking([rd_model], [track_model], num_classes)
    optimizer = config.get_optimizer(optimizer_config, model, lr)
    lr_scheduler = config.get_lr_scheduler(lr_config, optimizer)
    criterion = config.get_criterion(loss_config)

    start_epoch = 0
    best_acc = 0.
    if resume:
        checkpoint = torch.load(resume, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and device == "cuda":
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        logger.log(f"Loaded checkpoint from {resume}")
    model.to(device)

    train_transform, val_transform = config.get_transform(image_channels, height, width)

    train_batch_files, val_batch_files = split_train_val(data_root, num_classes, val_ratio, shuffle)
    train_dataset = dataset.FusedDataset(train_batch_files, image_transform=train_transform, image_seq_len=image_seq_len,
                                         track_seq_len=track_seq_len)
    val_dataset = dataset.FusedDataset(val_batch_files, image_transform=val_transform, image_seq_len=image_seq_len,
                                       track_seq_len=track_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              collate_fn=dataset.FusedDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=dataset.FusedDataset.collate_fn)


    for epoch in range(start_epoch, epochs):
        logger.log(f'-------------- Epoch {epoch + 1}/{epochs} --------------')

        train_accuracies, train_loss, train_acc, train_time, train_max_begin_time, train_avg_rate, train_conf_mat = \
            train(model, train_loader, optimizer, criterion, device, num_classes, track_seq_len, use_flash_attn)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/avg_acc", train_acc, epoch)
        writer.add_scalar("train/avg_begin_time", train_max_begin_time, epoch)
        writer.add_scalar("train/avg_rate", train_avg_rate, epoch)
        logger.log(f"Train Loss: {train_loss:.3f}\n"
                   f"Train Accuracy: {train_acc:.3f}\n"
                   f"Train Time: {time.time() - train_time:.3f}s\n"
                   f"Train Avg Begin Time: {train_max_begin_time}\n"
                   f"Train Avg Rate: {train_avg_rate:.3f}\n")
        if epoch % 10 == 0:
            logger.log(f"Confusion Matrix:\n{train_conf_mat}")
        for i, acc in enumerate(train_accuracies):
            writer.add_scalar(f"train/acc_{i}", acc, epoch)

        val_accuracies, val_loss, val_acc, val_time, val_max_begin_time, val_avg_rate, val_conf_mat = \
            val(model, val_loader, criterion, device, num_classes, track_seq_len, use_flash_attn)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/avg_acc", val_acc, epoch)
        writer.add_scalar("val/avg_begin_time", val_max_begin_time, epoch)
        writer.add_scalar("val/avg_rate", val_avg_rate, epoch)
        logger.log(f"Val Loss: {val_loss:.3f}\n"
                   f"Val Accuracy: {val_acc:.3f}\n"
                   f"Val Time: {time.time() - val_time:.3f}s\n"
                   f"Val Avg Begin Time: {val_max_begin_time}\n"
                   f"Val Avg Rate: {val_avg_rate:.3f}")
        if epoch % 10 == 0:
            logger.log(f"Confusion Matrix:\n{val_conf_mat}")
        for i, acc in enumerate(val_accuracies):
            writer.add_scalar(f"val/acc_{i}", acc, epoch)

        lr_scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, optimizer, lr_scheduler, epoch, best_acc, os.path.join(log_path, "best.pth"))
            logger.log(f"Best model saved with acc: {best_acc:.3f}")
        save_model(model, optimizer, lr_scheduler, epoch, best_acc, os.path.join(log_path, "latest.pth"))
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
    logger.log(f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    if result_path:
        config.check_paths(result_path)
        logger.log(f"Testing on train set...")
        model.load_state_dict(torch.load(os.path.join(log_path, "best.pth"), weights_only=False)['state_dict'])

        train_acc, train_avg_rate = test(model, train_loader, device, num_classes, track_seq_len,
                                         os.path.join(result_path, "train"), use_flash_attn)
        logger.log(f"Train Accuracy: {train_acc:.3f}\n"
                   f"Train Avg Rate: {train_avg_rate:.3f}")

        logger.log(f"Testing on val set...")
        val_acc, val_avg_rate = test(model, val_loader, device, num_classes, track_seq_len,
                                     os.path.join(result_path, "val"), use_flash_attn)
        logger.log(f"Val Accuracy: {val_acc:.3f}\n"
                   f"Val Avg Rate: {val_avg_rate:.3f}")

    logger.close()
    writer.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
