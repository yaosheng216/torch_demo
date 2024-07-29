import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBase(nn.Module):

    def __init__(self):
        super(VGGBase, self).__init__()
        # 3 * 28 * 28 (crop --> 32, 28)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 14 * 14
        self.conv2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())

        self.conv2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 7 * 7
        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU())

        self.conv3_2 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU())
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # 4 * 4
        self.conv4_1 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU())

        self.conv4_2 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU())
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2 * 2
        # self.conv5_1 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
        #                              nn.BatchNorm2d(1024),
        #                              nn.ReLU())
        #
        # self.conv5_2 = nn.Sequential(nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
        #                              nn.BatchNorm2d(1024),
        #                              nn.ReLU())
        # self.max_pooling5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # batchsize = 512 * 2 * 2 --> batchsize * (512 * 4)
        self.fc = nn.Linear(512 * 4, 10)

    def forward(self, x):
        batchsize = x.size(0)
        out = self.conv1(x)
        out = self.max_pooling1(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.max_pooling3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.max_pooling4(out)

        out = out.view(batchsize, -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out


def VGGNet():
    return VGGBase()
