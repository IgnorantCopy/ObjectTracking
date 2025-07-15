import argparse
import os
import time
from datetime import datetime
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


def train(model, train_loader, optimizer, criterion, device, logger, writer, epoch, num_classes, use_flash_attn):
    model.train()
    train_loss = 0.
    train_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    for i, (_, image, track_features, image_mask, track_mask, label) in enumerate(train_loader):
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
        optimizer.zero_grad()
        output = model(track_features, image, track_mask, image_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = output.max(1)
        totals += np.bincount(label.cpu().numpy(), minlength=num_classes)
        for j in range(num_classes):
            corrects[j] += (pred[label == j] == j).float().sum().item()
    train_accuracies = corrects / totals
    train_acc = corrects.sum() / totals.sum()
    train_loss = train_loss / len(train_loader)
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/avg_acc", train_acc, epoch)
    logger.log(
        f"Epoch {epoch + 1}:\n\tTrain Loss: {train_loss:.3f}\n\tTrain Accuracy: {train_acc:.3f}\n\tTrain Time: {time.time() - train_time:.3f}s")
    for i, acc in enumerate(train_accuracies):
        writer.add_scalar(f"train/acc_{i}", acc, epoch)


def val(model, val_loader, criterion, device, logger, writer, epoch, num_classes, use_flash_attn):
    model.eval()
    val_loss = 0.
    val_time = time.time()
    totals = np.array([0. for _ in range(num_classes)])
    corrects = np.array([0. for _ in range(num_classes)])
    with torch.no_grad():
        for i, (_, image, track_features, image_mask, track_mask, label) in enumerate(val_loader):
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
            output = model(track_features, image, track_mask, image_mask)
            loss = criterion(output, label)
            val_loss += loss.item()
            _, pred = output.max(1)
            totals += np.bincount(label.cpu().numpy(), minlength=num_classes)
            for j in range(num_classes):
                corrects[j] += (pred[label == j] == j).float().sum().item()
    val_accuracies = corrects / totals
    val_acc = corrects.sum() / totals.sum()
    val_loss = val_loss / len(val_loader)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/avg_acc", val_acc, epoch)
    logger.log(
        f"\n\tVal Loss: {val_loss:.3f}\n\tVal Accuracy: {val_acc:.3f}\n\tVal Time: {time.time() - val_time:.3f}s")
    for i, acc in enumerate(val_accuracies):
        writer.add_scalar(f"val/acc_{i}", acc, epoch)
    return val_loss, val_acc


def test(model, train_loader, val_loader, device, logger, result_path, log_path, use_flash_attn):
    logger.log("Start test on train set...")
    model.load_state_dict(torch.load(os.path.join(log_path, "best.pth"), weights_only=False)['state_dict'])
    model.eval()
    for i, (batch_files, image, track_features, image_mask, track_mask, label) in enumerate(train_loader):
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
        output = model(track_features, image, track_mask, image_mask)
        _, pred = output.max(1)
        label = label.cpu().numpy()
        pred = pred.cpu().numpy()
        for batch in range(len(batch_files)):
            batch_file = batch_files[batch]
            rd_matrices, ranges, velocities = dataset.process_batch(batch_file)
            batch_image_mask = image_mask[batch].cpu().numpy()
            if len(rd_matrices) > len(batch_image_mask):
                rd_matrices = rd_matrices[:len(batch_image_mask)]
                ranges = ranges[:len(batch_image_mask)]
                velocities = velocities[:len(batch_image_mask)]
            for frame in range(len(batch_image_mask)):
                if not batch_image_mask[frame]:
                    break
                frame_image = rd_matrices[frame]
                range_axis = ranges[frame]
                velocity_axis = velocities[frame]
                cls = batch_file.label
                if label[batch] == pred[batch]:
                    file_dir = os.path.join(result_path, f"train/correct/Label_{cls}/Batch_{batch_file.batch_num}/Frame_{frame + 1}")
                else:
                    file_dir = os.path.join(result_path, f"train/wrong/Label_{cls}/Batch_{batch_file.batch_num}/Frame_{frame + 1}")
                os.makedirs(file_dir, exist_ok=True)
                visualize.visualize_rd_matrix(frame_image, range_axis, velocity_axis, batch_file.batch_num,
                                              pred[batch] + 1, frame + 1, save_path=os.path.join(file_dir, "rd_map.png"))
                visualize.plot_3d_trajectory(batch_file.point_file, save_path=os.path.join(file_dir, "trajectory.png"))
                print(f"{file_dir}/rd_map.png and {file_dir}/trajectory.png saved.")

    logger.log("Start test on val set...")
    for i, (batch_files, image, track_features, image_mask, track_mask, label) in enumerate(val_loader):
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
        output = model(track_features, image, track_mask, image_mask)
        _, pred = output.max(1)
        label = label.cpu().numpy()
        pred = pred.cpu().numpy()
        for batch in range(len(batch_files)):
            batch_file = batch_files[batch]
            rd_matrices, ranges, velocities = dataset.process_batch(batch_file)
            batch_image_mask = image_mask[batch].cpu().numpy()
            if len(rd_matrices) > len(batch_image_mask):
                rd_matrices = rd_matrices[:len(batch_image_mask)]
                ranges = ranges[:len(batch_image_mask)]
                velocities = velocities[:len(batch_image_mask)]
            for frame in range(len(batch_image_mask)):
                if not batch_image_mask[frame]:
                    break
                frame_image = rd_matrices[frame]
                range_axis = ranges[frame]
                velocity_axis = velocities[frame]
                cls = batch_file.label
                if label[batch] == pred[batch]:
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
        train(model, train_loader, optimizer, criterion, device, logger, writer, epoch, num_classes, use_flash_attn)
        val_loss, val_acc = val(model, val_loader, criterion, device, logger, writer, epoch, num_classes, use_flash_attn)

        lr_scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, optimizer, lr_scheduler, epoch, best_acc, os.path.join(log_path, "best.pth"))
            logger.log(f"Best model saved with acc: {best_acc:.3f}")
        save_model(model, optimizer, lr_scheduler, epoch, best_acc, os.path.join(log_path, "latest.pth"))
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        logger.log('-' * 50)
    logger.log(f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    if result_path:
        config.check_paths(result_path)
        test(model, train_loader, val_loader, device, logger, result_path, log_path, use_flash_attn)

    logger.close()
    writer.close()


if __name__ == '__main__':
    main()
