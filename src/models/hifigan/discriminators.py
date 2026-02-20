from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d(in_ch, out_ch, k, s=1, p=None, g=1):
    if p is None:
        p = (k - 1) // 2
    return nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p, groups=g)


class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm_f(_conv1d(1, 128, 15, 1)),
            norm_f(_conv1d(128, 128, 41, 2, p=20, g=4)),
            norm_f(_conv1d(128, 256, 41, 2, p=20, g=16)),
            norm_f(_conv1d(256, 512, 41, 4, p=20, g=16)),
            norm_f(_conv1d(512, 1024, 41, 4, p=20, g=16)),
            norm_f(_conv1d(1024, 1024, 41, 1, p=20, g=16)),
            norm_f(_conv1d(1024, 1024, 5, 1, p=2)),
        ])
        self.post = norm_f(_conv1d(1024, 1, 3, 1, p=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(use_spectral_norm=False),
            ScaleDiscriminator(use_spectral_norm=False),
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outs, fmaps = [], []
        for i, d in enumerate(self.discs):
            if i > 0:
                x = self.pooling(x)
            o, fm = d(x)
            outs.append(o)
            fmaps.append(fm)
        return outs, fmaps


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (5, 1), (1, 1), padding=(2, 0))),
        ])
        self.post = norm_f(nn.Conv2d(1024, 1, (3, 1), (1, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List[int] = None):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outs, fmaps = [], []
        for d in self.discs:
            o, fm = d(x)
            outs.append(o)
            fmaps.append(fm)
        return outs, fmaps
