import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def discriminator_loss(real_outputs: List[torch.Tensor], fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for dr, df in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((dr - 1.0) ** 2) + torch.mean(df ** 2)
    return loss


def generator_adversarial_loss(fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for df in fake_outputs:
        loss = loss + torch.mean((df - 1.0) ** 2)
    return loss


def feature_matching_loss(real_fmaps: List[List[torch.Tensor]], fake_fmaps: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = 0.0
    for r_fm, f_fm in zip(real_fmaps, fake_fmaps):
        for r, f in zip(r_fm, f_fm):
            loss = loss + F.l1_loss(f, r)
    return loss


def mel_l1_loss(mel_real: torch.Tensor, mel_fake: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(mel_fake, mel_real)
