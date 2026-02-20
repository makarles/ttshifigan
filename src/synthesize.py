import os
from dataclasses import dataclass, field

import hydra
import torch
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio.mel import MelConfig
from src.datasets.custom_dir_dataset import CustomDirDataset, CustomDirDatasetConfig
from src.models.hifigan.generator import GeneratorV2, HiFiGANV2Config


@dataclass
class CkptConfig:
    path: str = "outputs_phase1/last.pt"


@dataclass
class SynthesizeConfig:
    mode: str = "resynthesize"
    out_dir: str = "synth_outputs"
    batch_size: int = 1
    num_workers: int = 0
    device: str = "cuda"

    save_ref: bool = False
    suffix: str = "_gen"

    ckpt: CkptConfig = field(default_factory=CkptConfig)
    dataset: CustomDirDatasetConfig = field(default_factory=lambda: CustomDirDatasetConfig(root="data/ruslan", audio_dir="audio"))
    mel: MelConfig = field(
        default_factory=lambda: MelConfig(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            f_min=0,
            f_max=11025,
            center=True,
            power=1.0,
            mel_norm=None,
        )
    )


def _pick_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _vocode_batch(G: torch.nn.Module, mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    if mel.dim() == 3:
        mel = mel.to(device)
    elif mel.dim() == 4:
        mel = mel.squeeze(1).to(device)
    else:
        raise RuntimeError(f"Unexpected mel shape: {tuple(mel.shape)}")

    y = G(mel)
    if y.dim() == 3:
        y = y.squeeze(1)
    return y.float().cpu()


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _copy_ref_wav(wav_path: str, out_path: str):
    wav, sr = torchaudio.load(wav_path)
    torchaudio.save(out_path, wav, sr)


@hydra.main(config_path="../configs", config_name="synthesize", version_base=None)
def main(cfg: SynthesizeConfig):
    print(OmegaConf.to_yaml(cfg))

    device = _pick_device(cfg.device)

    if cfg.mode not in ("resynthesize", "full_tts"):
        raise ValueError(f"Unknown mode: {cfg.mode}")

    if cfg.mode == "full_tts":
        raise NotImplementedError("full_tts mode is optional and not implemented in this script.")

    _safe_mkdir(cfg.out_dir)

    ds = CustomDirDataset(cfg.dataset, cfg.mel)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: batch,  # keep list of dicts
    )

    g_cfg = HiFiGANV2Config(n_mels=cfg.mel.n_mels)
    G = GeneratorV2(g_cfg).to(device)
    ckpt = torch.load(cfg.ckpt.path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    sr = cfg.mel.sample_rate

    for batch in tqdm(dl, desc="Synthesize"):
        mels = [b["mel"] for b in batch]
        names = [b["name"] for b in batch]
        wav_paths = [b["wav_path"] for b in batch]

        mel = torch.stack(mels, dim=0)
        y = _vocode_batch(G, mel, device=device)

        for i in range(y.size(0)):
            name = names[i]
            out_wav = os.path.join(cfg.out_dir, f"{name}{cfg.suffix}.wav")
            torchaudio.save(out_wav, y[i].unsqueeze(0), sample_rate=sr)

            if cfg.save_ref:
                out_ref = os.path.join(cfg.out_dir, f"{name}_ref.wav")
                _copy_ref_wav(wav_paths[i], out_ref)

    print("Done.")


if __name__ == "__main__":
    main()
