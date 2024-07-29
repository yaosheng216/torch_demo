import torch
import torchvision
import numpy as np


# 直接初始化定义
a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.type())

# 定义一个2x3的tensor
b = torch.Tensor(2, 3)
print(b)

# 定义一个全是1的3x3的tensor
c = torch.ones(3, 3)
c = torch.zeros_like(c)
print(c)

d = torch.eye(4, 4)
print(d)

# 生成一个随机的5x5的tensor
e = torch.rand(5, 5)
print(e)

f = torch.normal(mean=0.0, std=torch.rand(5))
print(f)

# 定义一个2x2的范围在-1到1的tensor
g = torch.Tensor(2, 2).uniform_(-1, 1)
print(g)

# 序列一个tensor
h = torch.arange(0, 11, 3)

# 创建一个等间隔的tensor
x = torch.linspace(2, 10,4)
print(x)

n = np.array([[1, 2], [3, 4]])
print(n)

m = np.zeros
print(m)

data = torch.tensor([1, 2, 3], dtype=torch.float32, device=torch.device(0))
# 稀疏tensor
# 稀疏表示tensor中非0元素的个数(coo)，0元素越多代表tensor越稀疏
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)
val = torch.sparse_coo_tensor(indices, values, [2, 4])

if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)   # 在GPU上创建tensor
    x = x.to(device)
    z = x + y
    print(z.to('cpu', torch.double))
