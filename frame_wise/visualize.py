import argparse
import os
import torch
from torchvision import transforms
import numpy as np
from math import log10
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt

from utils import config, dataset


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/frame_wise/vit.yaml', help='config file')
    parser.add_argument("--device", type=str, default='cuda', help='device', choices=['cpu', 'cuda'])
    parser.add_argument("--label",  type=int, default=2, help='label index')
    parser.add_argument("--batch",  type=int, default=1333, help='batch index')
    parser.add_argument("--frame",  type=int, default=0, help='frame index')
    parser.add_argument("--ckpt",   type=str, required=True, help='checkpoint file')
    args = parser.parse_args()
    return args


def _db(real, imag, eps=1e-10):
    real_mantissa, real_exponent = dataset.split_scientific_str(real)
    imag_mantissa, imag_exponent = dataset.split_scientific_str(imag)
    if real_exponent > imag_exponent:
        real_mantissa *= 10 ** (real_exponent - imag_exponent)
        return 10 * (log10(real_mantissa ** 2 + imag_mantissa ** 2 + eps) + 2 * imag_exponent)
    elif real_exponent < imag_exponent:
        imag_mantissa *= 10 ** (imag_exponent - real_exponent)
        return 10 * (log10(real_mantissa ** 2 + imag_mantissa ** 2 + eps) + 2 * real_exponent)
    else:
        return 10 * (log10(real_mantissa ** 2 + imag_mantissa ** 2 + eps) + 2 * real_exponent)

def _load_image(path):
    data = dataset.read_mat(path)
    rd_matrix = data['rd_matrix']
    velocity_axis = data['velocity_axis']
    velocity_mask = np.reshape(np.abs(velocity_axis) < 56, -1)
    rd_matrix = rd_matrix[:, velocity_mask]
    value = np.zeros_like(rd_matrix, dtype=np.float64)
    for i in range(len(rd_matrix)):
        for j in range(len(rd_matrix[i])):
            real = rd_matrix[i][j][0]
            imag = rd_matrix[i][j][1]
            value[i][j] = _db(real, imag)
    velocity_index = np.where(np.reshape(velocity_axis, -1) == 0)[0][0]
    value[:, velocity_index - 4:velocity_index + 3] = 0
    value[value < np.percentile(value, 5)] = 0
    return value[:, :, None]


def main():
    args = config_parser()
    config_path = args.config
    device = args.device if torch.cuda.is_available() else 'cpu'
    label = args.label
    batch = args.batch
    frame = args.frame
    model_path = args.ckpt

    model_config, train_config, data_config = config.get_config(config_path)

    data_root   = data_config['data_root']

    num_classes = train_config['num_classes']
    channels    = train_config['channels']
    height      = train_config['height']
    width       = train_config['width']

    model = config.get_model(model_config, channels, num_classes, height, width)
    params = torch.load(model_path)
    model.load_state_dict(params['state_dict'])
    model.to(device)
    target_layers = [model.transformer]
    cam = GradCAM(model, target_layers)

    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5 for _ in range(channels)],
                             std=[0.5 for _ in range(channels)]),
    ])
    image = _load_image(os.path.join(data_root, f"Label_{label}/Batch_{batch}/Frame_{frame}/MTD_result.m"))
    image = (np.max(image) - image) / (np.max(image) - np.min(image))
    image_tensor = to_tensor(image).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(label - 1)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=False)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

