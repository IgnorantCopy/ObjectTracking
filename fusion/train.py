import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils import config, dataset, visualize
from utils.logger import Logger


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default='./configs/fused.yaml', help="path to config file")
    parser.add_argument("--log-path",    type=str, default="./logs",               help="path to log file")
    parser.add_argument("--resume",      type=str, default=None,                   help="path to checkpoint file")
    parser.add_argument("--device",      type=str, default="cuda",                 help="device to use", choices=["cuda", "cpu"])
    parser.add_argument("--phase",       type=str, default="pretrain",             help="train or pretrain", choices=["train", "pretrain"])
    parser.add_argument("--pretrain",    type=str, default=None,                   help="path to the pretrain model")
    parser.add_argument("--result-path", type=str, default=None,                   help="path to store the result file")
    args = parser.parse_args()
    print(args)
    return args


def train(model, train_loader, optimizer, criterion, alpha, device, num_classes, track_seq_len, use_flash_attn):
    model.train()
    train_loss = 0.
    train_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    train_max_begin_time = 0
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
        begin_times = []
        pred = []
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
            pred_copy[output_max_t < 0.5] = 0

            begin = False
            for j in range(len(pred_copy)):
                if pred_copy[j] != 0 and not begin:
                    begin = True
                    begin_times.append(j)
                elif pred_copy[j] == 0 and begin:
                    pred_copy[j] = pred_t[j]
            if not begin:
                begin_times.append(len(pred_copy))
            pred.append(pred_copy.cpu().tolist())

            cls_loss_t = criterion(output_t, label)
            cls_loss += cls_loss_t
            # 释放显存
            del output_t, output_max_t, pred_t, pred_copy, index_mask_t, image_mask_t, track_mask_t
            torch.cuda.empty_cache()

        cls_loss /= track_seq_len
        begin_times = torch.tensor(begin_times)
        train_max_begin_time = max(train_max_begin_time, begin_times.max().item())
        time_loss = nn.MSELoss()(begin_times.float().to(device), torch.zeros_like(begin_times).float().to(device))
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
            unique_vals, counts = np.unique(pred_j, return_counts=True)
            pred_label = unique_vals[counts.argmax()]
            if pred_label == gt:
                corrects[gt] += 1
    train_accuracies = corrects / totals
    train_acc = corrects.sum() / totals.sum()
    train_loss = train_loss / len(train_loader)
    return train_accuracies, train_loss, train_acc, train_time, train_max_begin_time


def val(model, val_loader, criterion, alpha, device, num_classes, track_seq_len, use_flash_attn):
    model.eval()
    val_loss = 0.
    val_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    val_max_begin_time = 0
    with torch.no_grad():
        for i, (_, point_index, image, track_features, image_mask, track_mask, label) in tqdm(enumerate(val_loader), desc="Validation"):
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
            begin_times = []
            pred = []
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
                pred_copy[output_max_t < 0.5] = 0

                begin = False
                for j in range(len(pred_copy)):
                    if pred_copy[j] != 0 and not begin:
                        begin = True
                        begin_times.append(j)
                    elif pred_copy[j] == 0 and begin:
                        pred_copy[j] = pred_t[j]
                if not begin:
                    begin_times.append(len(pred_copy))
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
                unique_vals, counts = np.unique(pred_j, return_counts=True)
                pred_label = unique_vals[counts.argmax()]
                if pred_label == gt:
                    corrects[gt] += 1
    val_accuracies = corrects / totals
    val_acc = corrects.sum() / totals.sum()
    val_loss = val_loss / len(val_loader)

    return val_accuracies, val_loss, val_acc, val_time, val_max_begin_time


def test(model, train_loader, val_loader, device, track_seq_len, logger, result_path, log_path, use_flash_attn):
    logger.log("Start test on train set...")
    model.load_state_dict(torch.load(os.path.join(log_path, "best.pth"), weights_only=False)['state_dict'])
    model.eval()
    with (torch.no_grad()):
        for i, (batch_files, point_index, image, track_features, image_mask, track_mask, label) \
                in tqdm(enumerate(train_loader), total=len(train_loader), desc="Test on train set"):
            image = image.to(device)
            track_features = track_features.to(device)
            if use_flash_attn:
                image = image.half()
                track_features = track_features.half()
            else:
                image = image.float()
                track_features = track_features.float()
            image_mask = image_mask.to(device)
            track_mask = track_mask.to(device)
            label = label.to(device)

            pred = []
            for t in range(track_seq_len):
                index_mask_t = (point_index <= t + 1).to(device)
                image_t = image * index_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                track_features_t = track_features[:, :, :t+1, :]
                image_mask_t = image_mask * index_mask_t
                track_mask_t = track_mask[:, :t+1]
                output_t = model(track_features_t, image_t, track_mask_t, image_mask_t)
                output_max_t, pred_t = output_t.max(1)
                pred_copy = pred_t.clone()
                pred_copy[output_max_t < 0.5] = 0

                begin = False
                for j in range(len(pred_copy)):
                    if pred_copy[j] != 0 and not begin:
                        begin = True
                    elif pred_copy[j] == 0 and begin:
                        pred_copy[j] = pred_t[j]
                pred.append(pred_copy.cpu().tolist())

                del output_t, output_max_t, pred_t, pred_copy, index_mask_t, image_mask_t, track_mask_t
                torch.cuda.empty_cache()

            pred = np.array(pred).T   # [batch_size, seq_len]
            label = label.cpu().numpy()
            for batch in range(len(batch_files)):
                batch_file = batch_files[batch]
                rd_matrices, ranges, velocities = dataset.process_batch(batch_file)
                batch_image_mask = image_mask[batch].cpu().numpy()
                if len(rd_matrices) > len(batch_image_mask):
                    rd_matrices = rd_matrices[:len(batch_image_mask)]
                    ranges = ranges[:len(batch_image_mask)]
                    velocities = velocities[:len(batch_image_mask)]

                cls = batch_file.label
                batch_pred = pred[batch]
                unique_vals, counts = np.unique(batch_pred, return_counts=True)
                pred_label = unique_vals[counts.argmax()]

                for frame in range(len(batch_image_mask)):
                    if not batch_image_mask[frame]:
                        break
                    frame_image = rd_matrices[frame]
                    range_axis = ranges[frame]
                    velocity_axis = velocities[frame]

                    if label[batch] == pred_label:
                        file_dir = os.path.join(result_path, f"train/correct/Label_{cls}/Batch_{batch_file.batch_num}/Frame_{frame + 1}")
                    else:
                        file_dir = os.path.join(result_path, f"train/wrong/Label_{cls}/Batch_{batch_file.batch_num}/Frame_{frame + 1}")
                    os.makedirs(file_dir, exist_ok=True)

                    visualize.visualize_rd_matrix(frame_image, range_axis, velocity_axis, batch_file.batch_num,
                                                  pred[batch] + 1, frame + 1, save_path=os.path.join(file_dir, "rd_map.png"))
                    visualize.plot_3d_trajectory(batch_file.point_file, save_path=os.path.join(file_dir, "trajectory.png"))
                    print(f"{file_dir}/rd_map.png and {file_dir}/trajectory.png saved.")

    logger.log("Start test on val set...")
    with torch.no_grad():
        for i, (batch_files, point_index, image, track_features, image_mask, track_mask, label) \
                in tqdm(enumerate(val_loader), total=len(val_loader), desc="Test on val set"):
            image = image.to(device)
            track_features = track_features.to(device)
            if use_flash_attn:
                image = image.half()
                track_features = track_features.half()
            else:
                image = image.float()
                track_features = track_features.float()
            image_mask = image_mask.to(device)
            track_mask = track_mask.to(device)
            label = label.to(device)

            pred = []
            for t in tqdm(range(track_seq_len)):
                index_mask_t = (point_index <= t + 1).to(device)
                image_t = image * index_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                track_features_t = track_features[:, :, :t+1, :]
                image_mask_t = image_mask * index_mask_t
                track_mask_t = track_mask[:, :t+1]
                output_t = model(track_features_t, image_t, track_mask_t, image_mask_t)
                output_max_t, pred_t = output_t.max(1)
                pred_copy = pred_t.clone()
                pred_copy[output_max_t < 0.5] = 0

                begin = False
                for j in range(len(pred_copy)):
                    if pred_copy[j] != 0 and not begin:
                        begin = True
                    elif pred_copy[j] == 0 and begin:
                        pred_copy[j] = pred_t[j]
                pred.append(pred_copy.cpu().tolist())

                del output_t, output_max_t, pred_t, pred_copy, index_mask_t, image_mask_t, track_mask_t
                torch.cuda.empty_cache()

            pred = np.array(pred).T   # [batch_size, seq_len]
            label = label.cpu().numpy()
            for batch in range(len(batch_files)):
                batch_file = batch_files[batch]
                rd_matrices, ranges, velocities = dataset.process_batch(batch_file)
                batch_image_mask = image_mask[batch].cpu().numpy()
                if len(rd_matrices) > len(batch_image_mask):
                    rd_matrices = rd_matrices[:len(batch_image_mask)]
                    ranges = ranges[:len(batch_image_mask)]
                    velocities = velocities[:len(batch_image_mask)]

                cls = batch_file.label
                batch_pred = pred[batch]
                unique_vals, counts = np.unique(batch_pred, return_counts=True)
                pred_label = unique_vals[counts.argmax()]

                for frame in range(len(batch_image_mask)):
                    if not batch_image_mask[frame]:
                        break
                    frame_image = rd_matrices[frame]
                    range_axis = ranges[frame]
                    velocity_axis = velocities[frame]

                    if label[batch] == pred_label:
                        file_dir = os.path.join(result_path, f"val/correct/Label_{cls}/Batch_{batch_file.batch_num}/Frame_{frame + 1}")
                    else:
                        file_dir = os.path.join(result_path, f"val/wrong/Label_{cls}/Batch_{batch_file.batch_num}/Frame_{frame + 1}")
                    os.makedirs(file_dir, exist_ok=True)

                    visualize.visualize_rd_matrix(frame_image, range_axis, velocity_axis, batch_file.batch_num,
                                                  pred[batch] + 1, frame + 1, save_path=os.path.join(file_dir, "rd_map.png"))
                    visualize.plot_3d_trajectory(batch_file.point_file, save_path=os.path.join(file_dir, "trajectory.png"))
                    print(f"{file_dir}/rd_map.png and {file_dir}/trajectory.png saved.")


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
    config_path   = args.config_path
    log_path      = args.log_path
    resume        = args.resume
    device        = args.device if torch.cuda.is_available() else "cpu"
    phase         = args.phase
    pretrain      = args.pretrain
    result_path   = args.result_path

    log_path = os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    config.check_paths(log_path)
    writer = SummaryWriter(log_dir=log_path)
    logger = Logger(os.path.join(log_path, "train.txt"))
    rd_model_config, track_model_config, data_config, train_config = config.get_config(config_path)

    use_flash_attn = rd_model_config['name'] in ['Vit', 'ViViT']

    data_root     = data_config['data_root']
    val_ratio     = data_config['val_ratio']
    shuffle       = data_config['shuffle']
    image_seq_len = data_config['image_seq_len']
    track_seq_len = data_config['track_seq_len']
    config.check_paths(data_root)

    batch_size       = train_config['batch_size']
    num_workers      = train_config['num_workers']
    epochs           = train_config['num_epochs']
    lr               = train_config['lr']
    num_classes      = train_config['num_classes']
    channels         = train_config['channels']
    alpha            = train_config['alpha']
    height           = train_config['height']
    width            = train_config['width']
    lr_config        = train_config['lr_scheduler']
    optimizer_config = train_config['optimizer']
    loss_config      = train_config['loss']


    if phase == "pretrain":
        num_classes -= 2    # exclude noise and unknown class
        model = config.get_model(rd_model_config, track_model_config, channels, num_classes)
    elif phase == "train":
        if not pretrain:
            raise ValueError("Pretrain model is not provided.")
        pretrain_checkpoint = torch.load(pretrain)
        model = config.get_model(rd_model_config, track_model_config, channels, num_classes - 2)
        model.load_state_dict(pretrain_checkpoint['state_dict'])
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError("Invalid phase.")

    optimizer = config.get_optimizer(optimizer_config, model, lr)
    lr_scheduler = config.get_lr_scheduler(lr_config, optimizer)
    criterion = config.get_criterion(loss_config)

    start_epoch = 0
    best_acc = 0.
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and device == "cuda":
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Loaded checkpoint from {resume}")
        logger.log(f"Loaded checkpoint from {resume}")
    if use_flash_attn:
        model.half()
    model.to(device)

    train_transform, val_transform = config.get_transform(channels, height, width)
    train_batch_files, val_batch_files = dataset.split_train_val(data_root, num_classes, val_ratio, shuffle)
    train_dataset = dataset.FusedDataset(train_batch_files, image_transform=train_transform, image_seq_len=image_seq_len,
                                         track_seq_len=track_seq_len, track_transform=transforms.ToTensor())
    val_dataset = dataset.FusedDataset(val_batch_files, image_transform=val_transform, image_seq_len=image_seq_len,
                                       track_seq_len=track_seq_len, track_transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)

    for epoch in range(start_epoch, epochs):
        logger.log(f'-------------- Epoch {epoch + 1}/{epochs} --------------')

        train_accuracies, train_loss, train_acc, train_time, train_max_begin_time = \
            train(model, train_loader, optimizer, criterion, alpha, device, num_classes, track_seq_len, use_flash_attn)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/avg_acc", train_acc, epoch)
        writer.add_scalar("train/max_begin_time", train_max_begin_time, epoch)
        logger.log(f"Train Loss: {train_loss:.3f}\n"
                   f"Train Accuracy: {train_acc:.3f}\n"
                   f"Train Time: {time.time() - train_time:.3f}s\n"
                   f"Train Max Begin Time: {train_max_begin_time}\n")
        for i, acc in enumerate(train_accuracies):
            writer.add_scalar(f"train/acc_{i}", acc, epoch)

        val_accuracies, val_loss, val_acc, val_time, val_max_begin_time = \
            val(model, val_loader, criterion, alpha, device, num_classes, track_seq_len, use_flash_attn)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/avg_acc", val_acc, epoch)
        writer.add_scalar("val/max_begin_time", val_max_begin_time, epoch)
        logger.log(f"Val Loss: {val_loss:.3f}\n"
                   f"Val Accuracy: {val_acc:.3f}\n"
                   f"Val Time: {time.time() - val_time:.3f}s\n"
                   f"Val Max Begin Time: {val_max_begin_time}")
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
        test(model, train_loader, val_loader, device, track_seq_len, logger, result_path, log_path, use_flash_attn)

    logger.close()
    writer.close()


if __name__ == '__main__':
    main()
