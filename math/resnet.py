import torch
import torch.nn as nn
import torch.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.adjust_channels(x)
        out = out1 + out2
        return out


block = ResidualBlock(in_channels=64, out_channels=64)
input_tensor = torch.randn(1, 64, 32, 32)
output_tensor = block(input_tensor)
