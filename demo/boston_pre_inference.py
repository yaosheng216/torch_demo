import torch
import numpy as np
import re


class Net(torch.nn.Module):
    def __init__(self, feature, output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(feature, 100)
        self.predict = torch.nn.Linear(100, output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out

# 解析数据
ff = open('/Users/great/Desktop/housing.data.txt').readlines()
data = []
for item in ff:
    out = re.sub(r'\s{2,}', ' ', item).strip()
    print(out)
    data.append(out.split(' '))

data = np.array(data).astype(np.float32)
print(data.shape)

y = data[:, -1]
x = data[:, 0:-1]

x_train = x[0:496, ...]
y_train = y[0:496, ...]
x_test = x[496:, ...]
y_test = y[496:, ...]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

net = torch.load('model/model.pkl')
loss_func = torch.nn.MSELoss()
x_data = torch.tensor(x_test, dtype=torch.float32)
y_data = torch.tensor(y_test, dtype=torch.float32)
pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss_test = loss_func(pred, y_data) * 0.001
print('loss_train:{}'.format(loss_test))
