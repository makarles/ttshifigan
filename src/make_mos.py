import os
import argparse
import torch
import torchaudio

from src.audio.mel import MelConfig, MelExtractor, load_audio
from src.models.hifigan.generator import GeneratorV2, HiFiGANV2Config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_rate", type=int, default=22050)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    mel_cfg = MelConfig(
        sample_rate=args.sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=args.sample_rate // 2,
        center=True,
        power=1.0,
        mel_norm=None,
    )
    mel_extractor = MelExtractor(mel_cfg).to(device)

    g_cfg = HiFiGANV2Config(n_mels=mel_cfg.n_mels)
    G = GeneratorV2(g_cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    for name in ["1.wav", "2.wav", "3.wav"]:
        in_path = os.path.join(args.in_dir, name)
        if not os.path.isfile(in_path):
            raise FileNotFoundError(in_path)

        wav, sr = load_audio(in_path, target_sr=args.sample_rate)
        wav = wav.to(device)

        mel = mel_extractor(wav)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        y = G(mel).squeeze(0).squeeze(0).float().cpu()

        out_path = os.path.join(args.out_dir, name)
        torchaudio.save(out_path, y.unsqueeze(0), sample_rate=args.sample_rate)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
