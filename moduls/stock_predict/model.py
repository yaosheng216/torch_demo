import torch
import torch.nn as nn
import torch.nn.functional as F


class StockPredictNet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(StockPredictNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, 1)
        )

    def forward(self, x):
        out = self.hidden(x)
        return out

