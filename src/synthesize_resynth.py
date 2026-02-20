import os
import random
import torch
import torchaudio

from src.audio.mel import MelConfig, MelExtractor, load_audio
from src.models.hifigan.generator import GeneratorV2, HiFiGANV2Config


def crop_to_min_len(a: torch.Tensor, b: torch.Tensor):
    m = min(a.size(-1), b.size(-1))
    return a[..., :m], b[..., :m]


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "outputs_phase1/last.pt"
    in_dir = "data/ruslan/audio"
    out_dir = "resynth_outputs_phase2"
    os.makedirs(out_dir, exist_ok=True)

    mel_cfg = MelConfig(
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
    mel_extractor = MelExtractor(mel_cfg).to(device)

    g_cfg = HiFiGANV2Config(n_mels=mel_cfg.n_mels)
    G = GeneratorV2(g_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    files = [f for f in os.listdir(in_dir) if f.lower().endswith(".wav")]
    files = sorted(files)
    random.seed(42)
    picks = random.sample(files, k=min(20, len(files)))

    for fname in picks:
        path = os.path.join(in_dir, fname)
        wav, _ = load_audio(path, mel_cfg.sample_rate)
        wav = wav.to(device)

        mel = mel_extractor(wav).unsqueeze(0)
        wav_hat = G(mel)

        wav_ref, wav_gen = crop_to_min_len(wav, wav_hat[0])

        torchaudio.save(os.path.join(out_dir, fname.replace(".wav", "_gen.wav")),
                        wav_gen.cpu(), mel_cfg.sample_rate)

        torchaudio.save(os.path.join(out_dir, fname.replace(".wav", "_ref.wav")),
                        wav_ref.cpu(), mel_cfg.sample_rate)

    print(f"Saved {len(picks)} pairs to: {out_dir}")


if __name__ == "__main__":
    main()
