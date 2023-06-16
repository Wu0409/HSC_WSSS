import torch
from torch.nn import functional as F


def max_norm(p, e=1e-5):
    if p.dim() == 3:
        C, H, W = p.size()
        max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
        p = p / (max_v + e)
    elif p.dim() == 4:
        N, C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        p = F.relu(p - min_v - e) / (max_v - min_v + e)
    return p


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x