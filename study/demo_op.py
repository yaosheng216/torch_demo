import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

# print(a)
# print(b)
# print(a.add(b))
# print(a.sub(b))


c = torch.ones(2, 1)
d = torch.ones(1, 2)
# print(c)
# print(d)
# print(c @ d)
# print(c.matmul(d))
# print(torch.matmul(c, d))
# print(torch.mm(c, d))

# 高纬tensor
e = torch.ones(1, 2, 3, 4)
f = torch.ones(1, 2, 4, 3)
print(e)
print(f)
print(e.matmul(f))
print(e @ f)


# 指数运算
g = torch.tensor([1, 2])
# print(torch.pow(a, 3))
