import torch
import torch.nn as nn
import torch.nn.functional as F



class RDNet(nn.Module):
    def __init__(self, channels=1, num_classes=6):
        super(RDNet, self).__init__()
