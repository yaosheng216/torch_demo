import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tf
import os


class ImageDataset(Dataset):
    def __init__(self, root='', transform=None, model=None):
        self.transform = tf.Compose(transform)
        self.pathA = os.path.join(root, model, 'A/*')
        self.pathB = os.path.join(root, model, 'B/*')

        self.list_A = glob.glob(self.pathA)
        self.list_B = glob.glob(self.pathB)

    def __getitem__(self, index):
        im_pathA = self.list_A[index % len(self.list_A)]
        im_pathB = random.choice(self.list_B)

        im_A = Image.open(im_pathA)
        im_B = Image.open(im_pathB)

        item_A = self.transform(im_A)
        item_B = self.transform(im_B)

        return {'A': item_A, 'B': item_B}


if __name__ == 'main':
    from torch.utils.data import DataLoader

    root = 'dataset/apple2orange'
    transform_ = [tf.Resize(256, Image.BILINEAR)]
    dataLoader = DataLoader(ImageDataset(root, transform_, 'train'), shuffle=True, num_workers=1)

    for i, batch in enumerate(dataLoader):
        print(i)
        print(batch)
