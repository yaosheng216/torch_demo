import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import cv2


# data
train_data = dataset.MNIST(root='mnist', train=True, transform=transforms.ToTensor, download=True)
test_data = dataset.MNIST(root='mnist', train=False, transform=transforms.ToTensor, download=False)

# batch_size
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


cnn = torch.load('model/mnist_model.pkl')
cnn = cnn.cuda()

# loss
loss_func = torch.nn.CrossEntropyLoss()

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
    # tensor.max(1): 求tensor中第一维中的最大值索引
    _, pred = output.max(1)
    accuracy = (pred == labels).sum().item

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    # batch_size * 1 * 28 * 28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_labels = labels[idx]

        print('labels', labels)
        print('pred', pred[idx])
        im_data = im_data.transpose(1, 2, 0)
        cv2.imshow('im_data', im_data)
        cv2.waitKey(0)


accuracy = accuracy / len(test_data)
loss_test = loss_test / (len(test_data) // 64)
