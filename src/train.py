import os
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
import wandb

from src.utils.seed import set_seed
from src.utils.checkpointing import save_checkpoint, load_checkpoint
from src.audio.mel import MelConfig, MelExtractor
from src.datasets.ruslan_dataset import RuslanDataset, RuslanDatasetConfig
from src.models.hifigan.generator import GeneratorV2, HiFiGANV2Config
from src.models.hifigan.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.models.hifigan.losses import (
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
    mel_l1_loss,
)


@dataclass
class LossWeights:
    lambda_fm: float = 2.0
    lambda_mel: float = 45.0


def _to_device(batch, device: torch.device):
    mel, wav, name = batch
    return mel.to(device), wav.to(device), name


def _crop_to_min_len(a: torch.Tensor, b: torch.Tensor):
    m = min(a.size(-1), b.size(-1))
    return a[..., :m], b[..., :m]


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    mel_cfg = MelConfig(**cfg.mel)
    mel_extractor = MelExtractor(mel_cfg).to(device)

    ds_cfg = RuslanDatasetConfig(
        root=cfg.data.root,
        audio_dir=cfg.data.audio_dir,
        max_files=cfg.data.max_files,
        segment_size=cfg.training.segment_size,
    )
    full_ds = RuslanDataset(ds_cfg, mel_cfg)

    n_total = len(full_ds)
    n_train = int(n_total * float(cfg.data.train_split))
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(cfg.training.pin_memory),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=bool(cfg.training.pin_memory),
        drop_last=False,
    )

    g_cfg = HiFiGANV2Config(n_mels=mel_cfg.n_mels)
    G = GeneratorV2(g_cfg).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    opt_g = torch.optim.AdamW(
        G.parameters(),
        lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.betas),
        weight_decay=float(cfg.optim.weight_decay),
    )
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.betas),
        weight_decay=float(cfg.optim.weight_decay),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    weights = LossWeights()

    os.makedirs(cfg.training.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.out_dir, "last.pt")

    start_step = 0
    if os.path.isfile(ckpt_path):
        state = load_checkpoint(ckpt_path, map_location="cpu")
        G.load_state_dict(state["G"])
        mpd.load_state_dict(state["mpd"])
        msd.load_state_dict(state["msd"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state.get("step", 0))
        print(f"Resumed from {ckpt_path}, step={start_step}")

    wandb.init(
        project=str(cfg.wandb.project),
        entity=None if cfg.wandb.entity in (None, "null") else str(cfg.wandb.entity),
        name=str(cfg.wandb.name),
        mode=str(cfg.wandb.mode),
        tags=list(cfg.wandb.tags) if cfg.wandb.tags is not None else None,
        notes=str(cfg.wandb.notes) if cfg.wandb.notes is not None else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    G.train()
    mpd.train()
    msd.train()

    step = start_step
    max_steps = int(cfg.training.max_steps)

    train_iter = iter(train_loader)

    start_time = time.time()
    pbar = tqdm(total=max_steps, initial=step, dynamic_ncols=True)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        mel, wav, _ = _to_device(batch, device)

        with torch.cuda.amp.autocast(enabled=True):
            wav_hat = G(mel).detach()
            wav_d, wav_hat_d = _crop_to_min_len(wav, wav_hat)

            real_mpd, _ = mpd(wav_d)
            fake_mpd, _ = mpd(wav_hat_d)
            loss_d_mpd = discriminator_loss(real_mpd, fake_mpd)

            real_msd, _ = msd(wav_d)
            fake_msd, _ = msd(wav_hat_d)
            loss_d_msd = discriminator_loss(real_msd, fake_msd)

            loss_d = loss_d_mpd + loss_d_msd

        opt_d.zero_grad(set_to_none=True)
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        with torch.cuda.amp.autocast(enabled=True):
            wav_hat = G(mel)
            wav_g, wav_hat_g = _crop_to_min_len(wav, wav_hat)

            fake_mpd, fmap_f_mpd = mpd(wav_hat_g)
            real_mpd, fmap_r_mpd = mpd(wav_g.detach())
            fake_msd, fmap_f_msd = msd(wav_hat_g)
            real_msd, fmap_r_msd = msd(wav_g.detach())

            loss_g_adv = generator_adversarial_loss(fake_mpd) + generator_adversarial_loss(fake_msd)
            loss_fm = feature_matching_loss(fmap_r_mpd, fmap_f_mpd) + feature_matching_loss(fmap_r_msd, fmap_f_msd)

            mel_fake_list: List[torch.Tensor] = []
            for b in range(wav_hat_g.size(0)):
                mel_fake_list.append(mel_extractor(wav_hat_g[b]))
            mel_fake = torch.stack(mel_fake_list, dim=0)

            loss_mel = mel_l1_loss(mel, mel_fake)
            loss_g = loss_g_adv + weights.lambda_fm * loss_fm + weights.lambda_mel * loss_mel

        opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()

        elapsed = time.time() - start_time
        steps_done = step - start_step + 1
        steps_left = max_steps - step

        if steps_done > 0:
            sec_per_step = elapsed / steps_done
            eta_sec = steps_left * sec_per_step
            eta_hours = eta_sec / 3600
            steps_per_sec = 1.0 / sec_per_step
        else:
            eta_hours = 0.0
            steps_per_sec = 0.0

        wandb.log(
            {
                "loss/d_total": float(loss_d.item()),
                "loss/g_total": float(loss_g.item()),
                "loss/mel": float(loss_mel.item()),
                "eta_hours": eta_hours,
                "steps_per_sec": steps_per_sec,
                "step": step,
            },
            step=step,
        )

        step += 1
        pbar.update(1)
        pbar.set_postfix({
            "g_loss": f"{loss_g.item():.3f}",
            "d_loss": f"{loss_d.item():.3f}",
        })

    pbar.close()

    save_checkpoint(
        ckpt_path,
        {
            "step": step,
            "G": G.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "scaler": scaler.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )


if __name__ == "__main__":
    main()
