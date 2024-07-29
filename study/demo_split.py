import torch

a = torch.rand((3, 4))
print(a)
out = torch.chunk(a, 2, dim=1)
print(out)

split = torch.split(a, 3, dim=1)
print(split)
