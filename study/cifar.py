from torchvision import datasets
from torchvision import transforms

data_path = '/Users/great/Downloads/cache'

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=True, download=True)

print(len(cifar10))
print(cifar10.__getitem__(5))

img, label = cifar10[99]
dir(transforms)

to_tensor = transforms.ToTensor()
imq_t = to_tensor(img)


tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
