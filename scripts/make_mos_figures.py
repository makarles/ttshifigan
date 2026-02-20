import os
import glob
import argparse
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt


def load_mono(path: str):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav[0], sr


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int):
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)


def crop_to_min(a: torch.Tensor, b: torch.Tensor):
    m = min(a.numel(), b.numel())
    return a[:m], b[:m]


def stft_db(wav: torch.Tensor, n_fft=1024, hop_length=256, win_length=1024):
    window = torch.hann_window(win_length, device=wav.device)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-7)
    db = 20.0 * torch.log10(mag)
    return db.cpu().numpy()


def save_waveform(ref: np.ndarray, gen: np.ndarray, sr: int, out_path: str, title: str):
    t = np.arange(ref.shape[0]) / sr
    plt.figure()
    plt.plot(t, ref, label="ref")
    plt.plot(t, gen, label="gen", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_spec(db: np.ndarray, out_path: str, title: str):
    plt.figure()
    plt.imshow(db, origin="lower", aspect="auto")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True, help="Folder with MOS ground-truth wavs (1.wav,2.wav,3.wav)")
    ap.add_argument("--gen_dir", required=True, help="Folder with generated wavs with same names")
    ap.add_argument("--out_dir", required=True, help="Where to save figures")
    ap.add_argument("--target_sr", type=int, default=22050)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--win_length", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ref_paths = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
    if len(ref_paths) == 0:
        raise RuntimeError(f"No wavs found in ref_dir={args.ref_dir}")

    for ref_path in ref_paths:
        name = os.path.basename(ref_path)
        gen_path = os.path.join(args.gen_dir, name)
        if not os.path.isfile(gen_path):
            print(f"Skip (no gen): {name}")
            continue

        ref_wav, sr_ref = load_mono(ref_path)
        gen_wav, sr_gen = load_mono(gen_path)

        ref_wav = resample_if_needed(ref_wav, sr_ref, args.target_sr)
        gen_wav = resample_if_needed(gen_wav, sr_gen, args.target_sr)

        ref_wav, gen_wav = crop_to_min(ref_wav, gen_wav)

        base = os.path.splitext(name)[0]
        out_sub = os.path.join(args.out_dir, base)
        os.makedirs(out_sub, exist_ok=True)

        save_waveform(
            ref_wav.cpu().numpy(),
            gen_wav.cpu().numpy(),
            args.target_sr,
            os.path.join(out_sub, f"{base}_wave.png"),
            title=f"MOS {base}: waveform (ref vs gen) @ {args.target_sr} Hz",
        )

        ref_db = stft_db(ref_wav, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
        gen_db = stft_db(gen_wav, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)

        save_spec(ref_db, os.path.join(out_sub, f"{base}_stft_ref.png"), title=f"MOS {base}: STFT (ref, dB)")
        save_spec(gen_db, os.path.join(out_sub, f"{base}_stft_gen.png"), title=f"MOS {base}: STFT (gen, dB)")

        print(f"Saved: {base}")

    print(f"Done. Figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
