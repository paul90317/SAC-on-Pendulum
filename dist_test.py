import torch
from torch.distributions import Categorical
device = torch.device('cuda')
logits = torch.zeros((4, 2, 3), device=device)
dist = Categorical(logits=logits)
print(dist.entropy())
print(dist.sample())
print(dist.probs)

logits = torch.zeros((4, 2, 3), device=device)
index = torch.zeros((4, 2, 1), device=device, dtype=torch.int64)
print(logits.gather(-1, index))