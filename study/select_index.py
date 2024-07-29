import torch

a = torch.rand(4, 4)
b = torch.rand(4, 4)
print(a)
print(b)

out = torch.where(a > 0.5, a, b)
print(out)

# torch.index_select
idx = torch.index_select(a, dim=0, index=torch.tensor([0, 3, 2]))
print(idx)

c = torch.linspace(1, 16, 16).view(4, 4)

gath = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 1, 1],
                                                 [0, 1, 2, 2],
                                                  [0, 1, 3, 3]]))
print(gath)
print(gath.shape)

e = torch.linspace(1, 16, 16).view(4, 4)
print(e)
mask = torch.gt(a, 8)
print(mask)
torch.masked_select(a, mask)
