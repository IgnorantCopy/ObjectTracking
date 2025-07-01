import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        mid_channels = out_channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.shortcut else self.conv(x)


class CBSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbl = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbl(x)


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.bottleneck = nn.Sequential(
            *[BottleNeck(in_channels=mid_channels, out_channels=out_channels, shortcut=shortcut)
              for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.concat((self.bottleneck(x1), x2), dim=1)
        return self.conv3(x)


class RDNet(nn.Module):
    def __init__(self, height, width, channels=1, num_classes=6):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.cbs1 = CBSBlock(in_channels=channels, out_channels=64)
        self.cbs2 = CBSBlock(in_channels=64, out_channels=128)
        self.csp1 = CSPBlock(in_channels=128, out_channels=256, n=1, shortcut=True)

