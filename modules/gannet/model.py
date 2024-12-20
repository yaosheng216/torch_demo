import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        # nn.ReflectionPad1d:对一维输入进行反射填充,反射填充是指在输入张量的边缘使用其自身的反射进行填充
        # nn.InstanceNorm2d:实例化归一层
        # nn.ReLU(inplace=True):对输入张量应用ReLU激活函数，inplace=True 的作用是直接在输入张量上进行操作，而不返回新的张量，节省内存
        conv_block = [nn.ReflectionPad1d(1),
                      nn.Conv2d(in_channel, in_channel, 3),
                      nn.InstanceNorm2d(in_channel),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad1d(1),
                      nn.Conv2d(in_channel, in_channel, 3),
                      nn.InstanceNorm2d(in_channel)]
        # nn.Sequential:将多个神经网络层按顺序排列
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        net = [nn.ReflectionPad1d(3),
               nn.Conv2d(3, 64, 7),
               nn.InstanceNorm2d(64),
               nn.ReLU(inplace=True)]

        # 下采样
        in_channel = 64
        out_channel = in_channel * 2
        for _ in range(2):
            net += [nn.Conv2d(in_channel, out_channel, 7, stride=2, padding=1),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)]
            in_channel = out_channel
            out_channel = in_channel * 2

        for _ in range(9):
            net += [ResBlock(in_channel)]

        # 上采样
        out_channel = in_channel // 2
        for _ in range(2):
            # nn.ConvTranspose2d:用于实现二维的转置卷积（也称为反卷积或上采样卷积），转置卷积通常用于生成模型和自动编码器中，从低维度的特征图生成高维度的图像
            net += [nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=2),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)]
            in_channel = out_channel
            out_channel = in_channel // 2

        net += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh]
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        model = [nn.Conv2d(3, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


if __name__ == 'main':
    G = GeneratorNet()
    D = DiscriminatorNet()
    import torch
    input_tensor = torch.ones((1, 3, 256, 256), dtype=torch.float)
    out = G(input_tensor)
    print(out.size())

    out = D(input_tensor)
    print(out.size())
