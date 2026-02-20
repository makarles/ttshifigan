import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.audio.mel import MelExtractor, MelConfig, load_audio


@dataclass
class RuslanDatasetConfig:
    root: str
    audio_dir: str = "audio"
    max_files: Optional[int] = None
    segment_size: int = 8192


class RuslanDataset(Dataset):
    def __init__(self, cfg: RuslanDatasetConfig, mel_cfg: MelConfig):
        self.cfg = cfg
        self.mel_extractor = MelExtractor(mel_cfg)

        audio_path = os.path.join(cfg.root, cfg.audio_dir)
        if not os.path.isdir(audio_path):
            raise FileNotFoundError(f"Audio dir not found: {audio_path}")

        files = sorted([f for f in os.listdir(audio_path) if f.lower().endswith(".wav")])
        if cfg.max_files is not None:
            files = files[: cfg.max_files]

        self.paths: List[str] = [os.path.join(audio_path, f) for f in files]
        if len(self.paths) == 0:
            raise RuntimeError(f"No wav files found in {audio_path}")

    def __len__(self) -> int:
        return len(self.paths)

    def _random_segment(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.size(1)
        seg = self.cfg.segment_size
        if T < seg:
            pad = seg - T
            wav = torch.nn.functional.pad(wav, (0, pad))
            return wav
        start = random.randint(0, T - seg)
        return wav[:, start : start + seg]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[idx]
        wav, _ = load_audio(path, self.mel_extractor.cfg.sample_rate)
        wav_seg = self._random_segment(wav)
        mel = self.mel_extractor(wav_seg)
        name = os.path.splitext(os.path.basename(path))[0]
        return mel, wav_seg, name
