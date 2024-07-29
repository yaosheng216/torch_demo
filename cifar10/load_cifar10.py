from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert('RGB')


# train_transform = transforms.Compose([transforms.RandomResizedCrop((28, 28)),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.RandomVerticalFlip(),
#                                       transforms.RandomRotation(90),
#                                       transforms.RandomGrayscale(0.1),
#                                       transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#                                       transforms.ToTensor()
#                                       ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


class MyDataset(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            im_label_name = im_item.split('/')[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label, = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob('/Users/great/Downloads/dataset/CIFAR10/TRAIN/*/*.png')
im_test_list = glob.glob('/Users/great/Downloads/dataset/CIFAR10/TEST/*/*.png')

train_dataset = MyDataset(im_train_list, transform=train_transform)
test_dataset = MyDataset(im_test_list, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False, num_workers=4)
print('num_of_train', len(train_dataset))
print('num_of_test', len(test_dataset))
