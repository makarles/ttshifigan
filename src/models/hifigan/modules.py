import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        assert len(dilations) == 3
        self.convs = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size * d - d) // 2
            self.convs.append(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad))
        self.convs2 = nn.ModuleList()
        for d in [1, 1, 1]:
            pad = (kernel_size * d - d) // 2
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class MRF(nn.Module):
    def __init__(self, channels: int, kernels: List[int], dilation_sets: List[List[int]]):
        super().__init__()
        assert len(kernels) == len(dilation_sets)
        self.blocks = nn.ModuleList([
            ResBlock1(channels, k, dset) for k, dset in zip(kernels, dilation_sets)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for b in self.blocks:
            out = out + b(x)
        return out / len(self.blocks)
