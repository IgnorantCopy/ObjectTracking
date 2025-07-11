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

from utils import config, dataset
from utils.logger import Logger


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default='../configs/seq_wise/swin.yaml', help="path to config file")
    parser.add_argument("--log-path",    type=str, default="./logs",                       help="path to log file")
    parser.add_argument("--resume",      type=str, default=None,                           help="path to checkpoint file")
    parser.add_argument("--device",      type=str, default="cuda",                         help="device to use", choices=["cuda", "cpu"])
    parser.add_argument("--phase",       type=str, default="pretrain",                     help="train or pretrain", choices=["train", "pretrain"])
    parser.add_argument("--pretrain",    type=str, default=None,                           help="path to the pretrain model")
    args = parser.parse_args()
    print(args)
    return args


def cal_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = config_parser()
    config_path = args.config_path
    log_path    = args.log_path
    resume      = args.resume
    device      = args.device if torch.cuda.is_available() else "cpu"
    phase       = args.phase
    pretrain    = args.pretrain

    log_path = os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    config.check_paths(log_path)
    writer = SummaryWriter(log_dir=log_path)
    logger = Logger(os.path.join(log_path, "train.txt"))
    model_config, data_config, train_config = config.get_config(config_path)

    use_flash_attn = model_config['name'] in ['Vit', 'ViViT']

    data_root   = data_config['data_root']
    val_ratio   = data_config['val_ratio']
    shuffle     = data_config['shuffle']
    seq_len     = data_config['seq_len']
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

    start_epoch = 0
    best_acc = 0.

    if phase == "pretrain":
        num_classes -= 2    # exclude noise and unknown class
        model = config.get_model(model_config, channels, num_classes, height, width, seq_len)
    elif phase == "train":
        if not pretrain:
            raise ValueError("Pretrain model is not provided.")
        pretrain_checkpoint = torch.load(pretrain)
        model = config.get_model(model_config, channels, num_classes - 2, height, width, seq_len)
        model.load_state_dict(pretrain_checkpoint['state_dict'])
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError("Invalid phase.")

    optimizer = config.get_optimizer(optimizer_config, model, lr)
    lr_scheduler = config.get_lr_scheduler(lr_config, optimizer)
    criterion = config.get_criterion(loss_config)

    logger.log(f"Model has {cal_params(model) / 1024 ** 2:.2f}MB parameters.")
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
    train_paths, val_paths = dataset.split_train_val(data_root, num_classes, val_ratio, shuffle)
    train_dataset = dataset.RDSeq(train_paths, transform=train_transform, seq_len=seq_len)
    val_dataset = dataset.RDSeq(val_paths, transform=val_transform, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.
        train_time = time.time()
        totals = np.array([0. for _ in range(num_classes)])
        corrects = np.array([0. for _ in range(num_classes)])
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            if use_flash_attn:
                image = image.half()
            else:
                image = image.float()
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = output.max(1)
            totals += np.bincount(label.cpu().numpy(), minlength=num_classes)
            for j in range(num_classes):
                corrects[j] += (pred[label == j] == j).float().sum().item()
        train_accuracies =  corrects / totals
        train_acc = corrects.sum() / totals.sum()
        train_loss = train_loss / len(train_loader)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/avg_acc", train_acc, epoch)
        logger.log(f"Epoch {epoch+1}:\n\tTrain Loss: {train_loss:.3f}\n\tTrain Accuracy: {train_acc:.3f}\n\tTrain Time: {time.time()-train_time:.3f}s")
        for i, acc in enumerate(train_accuracies):
            writer.add_scalar(f"train/acc_{i}", acc, epoch)

        model.eval()
        val_loss = 0.
        val_time = time.time()
        totals = np.array([0. for _ in range(num_classes)])
        corrects = np.array([0. for _ in range(num_classes)])
        with torch.no_grad():
            for i, (image, label) in enumerate(val_loader):
                image = image.to(device)
                if use_flash_attn:
                    image = image.half()
                else:
                    image = image.float()
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label)
                val_loss += loss.item()
                _, pred = output.max(1)
                totals += np.bincount(label.cpu().numpy(), minlength=num_classes)
                for j in range(num_classes):
                    corrects[j] += (pred[label == j] == j).float().sum().item()
        val_accuracies =  corrects / totals
        val_acc = corrects.sum() / totals.sum()
        val_loss = val_loss / len(val_loader)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/avg_acc", val_acc, epoch)
        logger.log(f"\n\tVal Loss: {val_loss:.3f}\n\tVal Accuracy: {val_acc:.3f}\n\tVal Time: {time.time()-val_time:.3f}s")
        for i, acc in enumerate(val_accuracies):
            writer.add_scalar(f"val/acc_{i}", acc, epoch)

        lr_scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'best_acc': best_acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join(log_path, "best.pth"))
            logger.log(f"Best model saved with acc: {best_acc:.3f}")
        torch.save({
            'epoch': epoch+1,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, os.path.join(log_path, "latest.pth"))
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        logger.log('-' * 50)
    logger.log(f"Max GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    logger.close()
    writer.close()


if __name__ == '__main__':
    main()
