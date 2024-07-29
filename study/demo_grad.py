import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

v = Variable(torch.ones(2, 2))
x = torch.ones(2, 2, requires_grad=True)
x.register_hook(lambda grad: grad * 2)
print('x is:', x)
y = x + 2
print('y is:', y)
z = y * y * 3
print('z is:', z)
z.backward(torch.ones(2, 2))
print('backward is:', z)
print(x.grad)
print(y.grad)
print(torch.autograd.grad(z, [x, y]))

model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)

model1 = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(1, 20, 5)),
                        ('relu1', nn.ReLU()),
                        ('conv2', nn.Conv2d(20, 64, 5)),
                        ('relu2', nn.ReLU()),
                    ]))


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    # 前向推理
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

    # 反向推理
    def backward(self, y):
        y = F.tanh(self.forward(y))
        return F.relu(y)
    