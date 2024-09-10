import torch
import torch.nn as nn
import torch.nn.functional as F


class StockPredict(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(self, StockPredict).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.right = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * 2),
            nn.BatchNorm2d(in_channel * 2),
            nn.Tanh()
        )

    def forward(self, x):
        out  = self.left(x)
        out = self.right(out)
        out = F.softmax(out)
        out = F.relu(out, inplace=True)
        return out
