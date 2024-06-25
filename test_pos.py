import torch
from torch import nn
query_embed = nn.Embedding(100, 256)
print(query_embed.weight.shape)
query_embed, tgt = torch.split(query_embed.weight, 128, dim=1)
print(tgt.shape)