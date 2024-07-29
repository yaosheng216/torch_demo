import torch

a = torch.rand(2, 2)
print(a)
# 均值
print(torch.mean(a))
# 计算纬度均值
print(torch.mean(a, dim=0))
# 求和
print(torch.sum(a))
print(torch.sum(a, dim=0))
# 累积值
print(torch.prod(a))
print(torch.prod(a, dim=0))

# 最大值
print(torch.argmax(a, dim=0))
# 最小值
print(torch.argmin(a, dim=0))
# 标准差
print(torch.std(a))
print(torch.var(a))

# 中数
print(torch.median(a))
# 众位数
print(torch.mode(a))

# 直方图
b = torch.rand(2, 2) * 10
print(b)
print(torch.histc(a, 6, 0, 0))

# bincount
c = torch.randint(1, 10, [10])
print(c)
print(torch.bincount(c))
