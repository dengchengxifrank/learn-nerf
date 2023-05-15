import torch
a = torch.rand(2,4)

b = torch.chunk(a,4,dim=-1)

print(a,b)
