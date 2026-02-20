from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torchaudio


@dataclass
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 11025.0
    center: bool = True
    power: float = 1.0
    mel_norm: Optional[str] = None  # None / "slaney" etc


class MelExtractor(torch.nn.Module):
    def __init__(self, cfg: MelConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            center=cfg.center,
            power=cfg.power,
            norm=cfg.mel_norm,
            mel_scale="htk",
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power" if cfg.power == 2.0 else "magnitude")

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        if wav.dim() != 2 or wav.size(0) != 1:
            raise ValueError(f"Expected wav shape (1, T), got {tuple(wav.shape)}")
        mel = self.mel(wav)
        mel = mel.clamp_min(1e-5)
        mel_db = self.amp_to_db(mel)
        return mel_db.squeeze(0)


def load_audio(path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)  # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    wav = wav.clamp(-1.0, 1.0)
    return wav, sr