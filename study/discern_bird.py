import torch
import torch.nn as nn
import torch.cuda as cuda

n_out = 2
model = nn.Sequential(nn.Linear(3072, 512,),
                      nn.Tanh, nn.Linear(512, n_out))

data = torch.ones()
data.to(device=0)


alive = cuda.is_available().to_bytes()

