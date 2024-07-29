import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 10)

    def forward(self, x):
        out = self.model(x)
        return out


def resnet_18():
    return ResNet18()
