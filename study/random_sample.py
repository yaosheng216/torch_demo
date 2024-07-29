import torch

# 随机抽样结果检测（正态分布）
torch.manual_seed(1)   # 随机种子
mean = torch.rand(1, 2)
std = torch.rand(1, 2)
print(torch.normal(mean, std))

# 范数约束
a = torch.rand(2, 2) * 10
b = torch.rand(2, 2)

print(a)
print(torch.dist(a, b, p=1))
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=3))

print(torch.norm(a))
print(torch.norm(a, p=3))

# tensor裁剪
c = a.clamp(2, 5)
print(c)
