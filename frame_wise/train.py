import argparse
import os
from datetime import datetime
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils import config, dataset
from model import RDNet



def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",        type=str,   default='../configs/frame_wise.yaml',   help="path to config file")
    parser.add_argument("--log-path",           type=str,   default="./logs",                       help="path to log file")
    parser.add_argument("--resume",             type=str,   default=None,                           help="path to checkpoint file")
    parser.add_argument("--device",             type=str,   default="cuda",                         help="device to use", choices=["cuda", "cpu"])
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = config_parser()
    config_path     = args.config_path
    log_path        = args.log_path
    resume          = args.resume
    device          = args.device

    config.check_paths(config_path, log_path, resume)
    logger = SummaryWriter(os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S")))
    model_config, data_config, train_config = config.get_config(config_path)

    data_root   = data_config['data_root']
    val_ratio   = data_config['val_ratio']
    shuffle     = data_config['shuffle']

    batch_size      = train_config['batch_size']
    num_workers     = train_config['num_workers']
    epochs          = train_config['num_epochs']
    lr              = train_config['lr']
    momentum        = train_config['momentum']
    weight_decay    = train_config['weight_decay']
    channels        = train_config['channels']
    num_classes     = train_config['num_classes']
    height          = train_config['height']
    width           = train_config['width']
    lr_config       = train_config['lr_scheduler']
    optimizer_config= train_config['optimizer']
    loss_config     = train_config['loss']

    start_epoch = 0
    best_acc = 0.
    model = RDNet(channels=channels, num_classes=num_classes)
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Loaded checkpoint from {resume}")
    model.to(device)

    train_transform = transforms.Compose([

    ])
    val_transform = transforms.Compose([

    ])
    train_paths, val_paths = dataset.split_train_val(data_root, val_ratio, shuffle)
    train_dataset = dataset.RDMap(train_paths, transform=train_transform)
    val_dataset = dataset.RDMap(val_paths, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    optimizer = config.get_optimizer(optimizer_config, model, lr)
    lr_scheduler = config.get_lr_scheduler(lr_config, optimizer)
    criterion = config.get_criterion(loss_config)


