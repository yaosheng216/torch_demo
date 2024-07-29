import torch
import numpy as np

a = torch.rand(2, 1, 2, 3)
b = torch.rand(4, 2, 3)
c = a + b

print(a)
print(b)
print(c.shape)

d = torch.tensor([[1, 2, np.nan], [1, 2, 4]])
print(d)
