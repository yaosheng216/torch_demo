import torch

a = torch.rand(2, 3)
print(a)

out = torch.reshape(a, (3, 2))
print(out)
print(torch.t(out))
print(torch.transpose(out, 0, 1))
seq = torch.squeeze(a, (2, 1))

b = torch.full((2, 3), 10)
print(b)
