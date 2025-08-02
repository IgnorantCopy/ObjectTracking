import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing

from utils import config
from data import dataset


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default='./configs/swin.yaml', help="path to config file")
    parser.add_argument("--device",      type=str, default="cuda",                 help="device to use", choices=["cuda", "cpu"])
    parser.add_argument("--resume",      type=str, required=True,                  help="path to checkpoint file")
    parser.add_argument("--result-path", type=str, required=True,                  help="path to store the result file")
    args = parser.parse_args()
    print(args)
    return args


def test(model, train_loader, val_loader, device, threshold, track_seq_len, result_path, use_flash_attn):
    model.eval()
    with (torch.no_grad()):
        for i, (batch_files, point_index, image, extra_features, missing_rate, image_mask, label) \
                in tqdm(enumerate(train_loader), total=len(train_loader), desc="Test on train set"):
            image = image.to(device)
            if use_flash_attn:
                image = image.half()
            else:
                image = image.float()
            label = label.to(device)

            pred = []
            begin = [False for _ in range(len(image))]
            for t in range(track_seq_len):
                index_mask_t = (point_index <= t + 1)
                image_mask_t = image_mask * index_mask_t
                image_mask_t = image_mask_t.to(device)

                extra_features_t = []
                for j, mask in enumerate(image_mask_t):
                    rd_t = mask.float().sum().int()
                    if rd_t == 0:
                        extra_features_t.append(torch.tensor([0. for _ in range(extra_features.shape[2])]))
                    else:
                        extra_features_t.append(extra_features[j][:rd_t].mean(0))
                extra_features_t = torch.stack(extra_features_t).float().to(device)
                missing_rate_t = missing_rate[:, t].float().to(device)

                output_t = model(image, extra_features_t, missing_rate_t, image_mask_t)
                output_max_t, pred_t = output_t.max(1)
                pred_copy = pred_t.clone()
                pred_copy[output_max_t < threshold] = -1

                for j in range(len(pred_copy)):
                    if pred_copy[j] != -1 and not begin[j]:
                        begin[j] = True
                    elif pred_copy[j] == -1 and begin[j]:
                        pred_copy[j] = pred_t[j]
                pred.append(pred_copy.cpu().tolist())

                del output_t, output_max_t, pred_t, pred_copy, index_mask_t, image_mask_t
                torch.cuda.empty_cache()

            pred = np.array(pred).T   # [batch_size, seq_len]
            label = label.cpu().numpy()
            for batch in range(len(batch_files)):
                batch_file = batch_files[batch]
                cls = batch_file.label
                batch_pred = pred[batch]
                batch_pred = batch_pred[batch_pred != -1]
                if len(batch_pred) == 0:
                    pred_label = -1
                else:
                    unique_vals, counts = np.unique(batch_pred, return_counts=True)
                    pred_label = unique_vals[counts.argmax()]

                if label[batch] == pred_label:
                    file_dir = os.path.join(result_path, f"correct/Label_{cls}")
                else:
                    file_dir = os.path.join(result_path, f"wrong/Label_{cls}")
                os.makedirs(file_dir, exist_ok=True)
                np.savetxt(os.path.join(file_dir, f"Batch_{batch_file.batch_num}.txt"), pred[batch], fmt='%d')

    with torch.no_grad():
        for i, (batch_files, point_index, image, extra_features, missing_rate, image_mask, label) \
                in tqdm(enumerate(val_loader), total=len(val_loader), desc="Test on val set"):
            image = image.to(device)
            if use_flash_attn:
                image = image.half()
            else:
                image = image.float()
            label = label.to(device)

            pred = []
            begin = [False for _ in range(len(image))]
            for t in tqdm(range(track_seq_len)):
                index_mask_t = (point_index <= t + 1)
                image_mask_t = image_mask * index_mask_t
                image_mask_t = image_mask_t.to(device)

                extra_features_t = []
                for j, mask in enumerate(image_mask_t):
                    extra_features_t.append(extra_features[j][:mask.float().sum().int()].mean(0))
                extra_features_t = torch.stack(extra_features_t).float().to(device)
                missing_rate_t = missing_rate[:, t].float().to(device)

                output_t = model(image, extra_features_t, missing_rate_t, image_mask_t)
                output_max_t, pred_t = output_t.max(1)
                pred_copy = pred_t.clone()
                pred_copy[output_max_t < threshold] = -1

                for j in range(len(pred_copy)):
                    if pred_copy[j] != -1 and not begin[j]:
                        begin[j] = True
                    elif pred_copy[j] == -1 and begin[j]:
                        pred_copy[j] = pred_t[j]
                pred.append(pred_copy.cpu().tolist())

                del output_t, output_max_t, pred_t, pred_copy, index_mask_t, image_mask_t
                torch.cuda.empty_cache()

            pred = np.array(pred).T   # [batch_size, seq_len]
            label = label.cpu().numpy()
            for batch in range(len(batch_files)):
                batch_file = batch_files[batch]
                cls = batch_file.label
                batch_pred = pred[batch]
                batch_pred = batch_pred[batch_pred != -1]
                if len(batch_pred) == 0:
                    pred_label = -1
                else:
                    unique_vals, counts = np.unique(batch_pred, return_counts=True)
                    pred_label = unique_vals[counts.argmax()]

                if label[batch] == pred_label:
                    file_dir = os.path.join(result_path, f"correct/Label_{cls}")
                else:
                    file_dir = os.path.join(result_path, f"wrong/Label_{cls}")
                os.makedirs(file_dir, exist_ok=True)
                np.savetxt(os.path.join(file_dir, f"Batch_{batch_file.batch_num}.txt"), pred[batch], fmt='%d')


def main():
    args = config_parser()
    config_path   = args.config_path
    resume        = args.resume
    device        = args.device if torch.cuda.is_available() else "cpu"
    result_path   = args.result_path

    model_config, data_config, train_config = config.get_config(config_path)

    use_flash_attn = model_config['name'] in ['Vit', 'ViViT']

    data_root     = data_config['data_root']
    val_ratio     = data_config['val_ratio']
    shuffle       = data_config['shuffle']
    image_seq_len = data_config['image_seq_len']
    track_seq_len = data_config['track_seq_len']
    config.check_paths(data_root)

    batch_size       = train_config['batch_size']
    num_workers      = train_config['num_workers']
    lr               = train_config['lr']
    num_classes      = train_config['num_classes']
    channels         = train_config['channels']
    threshold        = train_config['threshold']
    height           = train_config['height']
    width            = train_config['width']
    lr_config        = train_config['lr_scheduler']
    optimizer_config = train_config['optimizer']


    num_classes -= 2    # exclude noise and unknown class
    model = config.get_model(model_config, channels, num_classes)

    optimizer = config.get_optimizer(optimizer_config, model, lr)
    lr_scheduler = config.get_lr_scheduler(lr_config, optimizer)
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and device == "cuda":
                    state[k] = v.cuda()
    else:
        raise ValueError(f"No checkpoint found at '{resume}'")
    if use_flash_attn:
        model.half()
    model.to(device)

    train_transform, val_transform = config.get_transform(channels, height, width)
    train_batch_files, val_batch_files = dataset.split_train_val(data_root, num_classes, val_ratio, shuffle)
    train_dataset = dataset.RDMap(train_batch_files, image_transform=train_transform, image_seq_len=image_seq_len,
                                  track_seq_len=track_seq_len, track_transform=transforms.ToTensor())
    val_dataset = dataset.RDMap(val_batch_files, image_transform=val_transform, image_seq_len=image_seq_len,
                                track_seq_len=track_seq_len, track_transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)

    if result_path:
        config.check_paths(result_path)
        test(model, train_loader, val_loader, device, threshold, track_seq_len, result_path, use_flash_attn)
    else:
        raise ValueError("No result path specified")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
