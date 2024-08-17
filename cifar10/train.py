import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from load_cifar10 import train_loader, test_loader
import os.path
import tensorboardX
from resnet import resnet
# from cifar10.mobilenet import mobilenet_small
from cifar10.inception import InceptionNetSmall

# 是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 训练轮数
epoch_num = 2000
# 学习率
lr = 0.01

batch_size = 128

net = InceptionNetSmall().to(device)

# loss
loss_func = nn.CrossEntropyLoss()

# 优化器（根据计算出的梯度更新每一层的权重和偏置）
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# 调整学习率（学习率优化策略）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

if not os.path.exists('logs'):
    os.mkdir('logs')
writer = tensorboardX.SummaryWriter('logs')

step_n = 0


if __name__ == '__main__':

    for epoch in range(epoch_num):
        print('epoch is:', epoch)
        net.train()  # train BN dropout

        for i, data in enumerate(train_loader):
            print('step is:', i)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('step is', i, 'loss is:', loss.item())

            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()
            print('train lr is:', optimizer.state_dict()['param_groups'][0]['lr'])
            print('train step', i, 'loss is:', loss.item(), 'mini-batch correct is:', 100.0 * correct / batch_size)

            writer.add_scalar('train loss', loss.item(), global_step=step_n)
            writer.add_scalar('train correct', 100.0 * correct.item() / batch_size, global_step=step_n)
            step_n += 1

            im = torchvision.utils.make_grid(inputs)
            writer.add_image('train im', im, global_step=step_n)

        if not os.path.exists('models'):
            os.mkdir('models')
        torch.save(net.state_dict(), 'models/{}.pth'.format(epoch + 1))
        scheduler.step()

        sum_loss = 0
        sum_correct = 0
        for i, data in enumerate(test_loader):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            # 损失函数计算
            loss = loss_func(outputs, labels)

            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()
            sum_loss += loss.item()
            sum_correct += correct.item()

            writer.add_scalar('test loss', loss.item(), global_step=step_n)
            writer.add_scalar('test correct', 100.0 * correct.item() / batch_size, global_step=step_n)

            im = torchvision.utils.make_grid(inputs)
            writer.add_image('test im', im, global_step=step_n)
            step_n += 1

        test_loss = sum_loss * 1.0 / len(test_loader)
        test_correct = sum_correct * 100.0 / len(test_loader) / batch_size
        print('epoch is', epoch + 1, 'loss is:', test_loss, 'test correct is:', test_correct)

writer.close()
