import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

# data
train_data = dataset.MNIST(root='mnist', train=True, transform=transforms.ToTensor, download=True)
test_data = dataset.MNIST(root='mnist', train=False, transform=transforms.ToTensor, download=False)

# batch_size
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


# model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=2, padding=2),
                                        torch.nn.BatchNorm2d(32),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2))
        # out_features: 输出特征值
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    # PyTorch中tensor的序列是: n * c * w
    def forward(self, x):
        out = self.conv(x)
        # 修改tensor的shape
        out = out.View(out.size()[0], -1)
        out = self.fc(out)
        return out


cnn = CNN()
cnn = cnn.cuda()
# loss
loss_func = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# training
for epoch in range(10000):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        output = cnn(images)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        loss.forward()
        optimizer.step()
        print('epoch is:{}, item is: {}/{}, loss is:{}'.format(
            epoch + 1, i, len(train_data) // 64, loss.item()))

    # test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()

        output = cnn(images)
        # [batch_size]
        # output = batch_size * cls_num
        loss_test += loss_func(output, labels)
        _, pred = output.max(1)
        accuracy = (pred == labels).sum().item

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    print('epoch is: {}, accuracy is: {}, loss test is: {}'.format(epoch + 1, accuracy, loss.item()))

# 存储模型
torch.save(cnn, 'model/mnist_model.pkl')
