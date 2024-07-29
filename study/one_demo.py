from torchvision import transforms
from PIL import Image
import torch


img = Image.open("/Users/great/Downloads/548080a3f73a01de38cdf437996005fc.jpg")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

print(batch_t)

with open('/Users/great/Downloads/image_class.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

a = torch.ones(3)
print(a[2])


some_list = list(range(6))
some_list[1:4]

double_torch = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.double)
float_torch = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
short_torch = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.short)

num_torch = torch.ones(10, 2).double()
# to()方法会检查转换是否有必要，若有必要，则执行转换
num1_torch = torch.ones(10, 2).to(dtype=torch.double)

a1 = torch.ones(3, 2)
at = torch.transpose(a1, 0, 1)

a1.shape
at.shape


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()