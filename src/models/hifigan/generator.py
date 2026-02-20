from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MRF, init_weights


@dataclass
class HiFiGANV2Config:
    # Mel
    n_mels: int = 80

    # Model
    upsample_rates: List[int] = None
    upsample_kernel_sizes: List[int] = None
    upsample_initial_channel: int = 128

    # MRF
    resblock_kernels: List[int] = None
    resblock_dilation_sets: List[List[int]] = None

    def __post_init__(self):
        # Common V2-ish lightweight config
        if self.upsample_rates is None:
            self.upsample_rates = [8, 8, 2, 2]  # total x256, hop=256 aligns
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 16, 4, 4]
        if self.resblock_kernels is None:
            self.resblock_kernels = [3, 5, 7]
        if self.resblock_dilation_sets is None:
            self.resblock_dilation_sets = [[1, 2, 4], [1, 2, 4], [1, 2, 4]]


class GeneratorV2(nn.Module):
    def __init__(self, cfg: HiFiGANV2Config):
        super().__init__()
        self.cfg = cfg

        self.pre = nn.Conv1d(cfg.n_mels, cfg.upsample_initial_channel, kernel_size=7, padding=3)

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        ch = cfg.upsample_initial_channel
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=k, stride=u, padding=(k - u) // 2)
            )
            ch = ch // 2
            self.mrfs.append(MRF(ch, cfg.resblock_kernels, cfg.resblock_dilation_sets))

        self.post = nn.Conv1d(ch, 1, kernel_size=7, padding=3)

        self.apply(init_weights)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        x = F.leaky_relu(x, 0.1)
        x = self.post(x)
        x = torch.tanh(x)
        return x
