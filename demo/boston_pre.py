import torch
import numpy as np
import re


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


# model
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


net = Net(13, 1)
# loss
loss_func = torch.nn.MSELoss()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

# training
for i in range(10000):
    x_data = torch.tensor(x_train, dtype=torch.float32)
    y_data = torch.tensor(y_train, dtype=torch.float32)

    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001
    print(pred)
    print(y_data.shape)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('item:{}, loss:{}'.format(i, loss))
    print(pred[0: 10])
    print(y_data[0: 10])

    # test
    x_data = torch.tensor(x_test, dtype=torch.float32)
    y_data = torch.tensor(y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print('item:{}, loss_train:{}'.format(i, loss_test))

# 存储模型
torch.save(net, 'model/model.pkl')
# torch.load()
# 只保存参数
# torch.save(net.state_dict(), 'params,pkl')
# net.load_state_dict()
