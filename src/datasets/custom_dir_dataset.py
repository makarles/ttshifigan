import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.audio.mel import MelConfig, MelExtractor, load_audio


@dataclass
class CustomDirDatasetConfig:
    root: str
    audio_dir: str = "audio"
    transcriptions_dir: str = "transcriptions"
    sample_rate: int = 22050
    use_transcriptions: bool = False


class CustomDirDataset(Dataset):

    def __init__(self, cfg: CustomDirDatasetConfig, mel_cfg: MelConfig):
        self.cfg = cfg
        self.mel_cfg = mel_cfg
        self.mel_extractor = MelExtractor(mel_cfg)

        self.audio_path = os.path.join(cfg.root, cfg.audio_dir)
        self.txt_path = os.path.join(cfg.root, cfg.transcriptions_dir)

        if not os.path.isdir(self.audio_path):
            raise FileNotFoundError(f"audio dir not found: {self.audio_path}")

        files = [f for f in os.listdir(self.audio_path) if f.lower().endswith(".wav")]
        files = sorted(files)
        if len(files) == 0:
            raise RuntimeError(f"No .wav files found in: {self.audio_path}")

        self.wav_files: List[str] = [os.path.join(self.audio_path, f) for f in files]

        self.txt_files: Optional[Dict[str, str]] = None
        if cfg.use_transcriptions:
            if not os.path.isdir(self.txt_path):
                raise FileNotFoundError(f"transcriptions dir not found: {self.txt_path}")
            txts = [f for f in os.listdir(self.txt_path) if f.lower().endswith(".txt")]
            mapping = {}
            for t in txts:
                stem = os.path.splitext(t)[0]
                mapping[stem] = os.path.join(self.txt_path, t)
            self.txt_files = mapping

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, idx: int) -> Dict:
        wav_path = self.wav_files[idx]
        base = os.path.splitext(os.path.basename(wav_path))[0]

        wav, _ = load_audio(wav_path, target_sr=self.cfg.sample_rate)  # (T,)
        mel = self.mel_extractor(wav)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        item = {
            "mel": mel,
            "name": base,
            "wav_path": wav_path,
        }

        if self.txt_files is not None:
            txt_path = self.txt_files.get(base)
            if txt_path is not None and os.path.isfile(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    item["text"] = f.read().strip()
            else:
                item["text"] = ""

        return item
