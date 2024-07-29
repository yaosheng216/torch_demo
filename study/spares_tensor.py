import torch

dev = torch.device('cpu')
dev1 = torch.device('cuda:0')

a = torch.tensor([2, 2], dtype=torch.float32, device=dev)
print(a)

i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
# 将稀疏的tensor转为紧密
x = torch.sparse_coo_tensor(i, v, [4, 4], dtype=torch.float32, device=dev).to_dense()
print(x)

b = torch.tensor([1, 2, 3], dtype=torch.float32, device=dev)

# 加法运算
c = a + b
d = torch.add(a, b)
a.add(b)
a.add_(b)

# 减法运算
e = a - b
f = torch.sub(a, b)
a.sub(b)
a.sub_(b)

# 乘法运算（哈达玛积:element wise，对应元素相乘）
g = a * b
h = torch.mul(a, b)
a.mul(b)
a.mul_(b)

# 除法运算
x = a / b
m = torch.div(a, b)
a.div(b)
a.div_(b)

# 幂运算
torch.exp(a)
a.exp()
a.exp_()

# 开方运算
a.sqrt()
a.sqrt_()

# 对数运算
torch.log(a)
torch.log2(a)
torch.log10(a)
torch.log_(a)