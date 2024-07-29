import torch

a = torch.zeros((2, 4))
print(a)
b = torch.ones((2, 4))
print(b)

out = torch.cat((a, b), dim=0)
print(out)

stack = torch.stack((a, b), dim=0)
print(stack)
