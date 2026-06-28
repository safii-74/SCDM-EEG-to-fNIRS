"""Train SCDM. Usage: PYTHONPATH=. python scripts/train.py --config configs/config.yaml

Supports resuming from a checkpoint. Set training.resume_from in config.yaml
to the checkpoint path, or pass --resume <path>.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SCDMDataset, collate
from src.data.hrf_features import compute_and_save as compute_hrf
from src.models.scdm import SCDM, DiffusionProcess
from src.training.trainer import Trainer


def load_coords(montage_path=None):
    if montage_path and Path(montage_path).exists():
        m = np.load(montage_path, allow_pickle=True)
        return m['eeg_coords'].item(), m['fnirs_coords'].item()
    from src.data.correlations import build_coords16
    rng = np.random.default_rng(0)
    return (build_coords16(rng.standard_normal((30, 2))),
            build_coords16(rng.standard_normal((36, 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--resume", default=None, help="Override resume checkpoint path")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dcfg, tcfg, mcfg = cfg["data"], cfg["training"], cfg["model"]
    device = tcfg["device"] if torch.cuda.is_available() else "cpu"

    use_hrf = mcfg.get("eeg_hrf", False)
    if use_hrf:
        hrf_path = dcfg.get("eeg_hrf_path", "data/preprocessed/eeg_hrf.npy")
        if Path(hrf_path).exists():
            print(f"Loading precomputed HRF features: {hrf_path}")
            eeg = np.load(hrf_path)
        else:
            print("Computing HRF-convolved EEG features (one-time)...")
            eeg = compute_hrf(dcfg["eeg_path"], hrf_path)
    else:
        eeg = np.load(dcfg["eeg_path"], mmap_mode='r')
    fnirs = np.load(dcfg["hbr_path"] if dcfg["modality"] == "hbr" else dcfg["hbo_path"])
    labels = np.load(dcfg["labels_path"])
    ec, fc = load_coords(dcfg.get("montage_path"))

    ds = SCDMDataset(eeg, fnirs, labels, ec, fc, cache_path=dcfg["planes_cache"])
    if ds._planes_mmap is None:
        print("Precomputing correlation planes (one-time)...")
        ds.precompute_planes()
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        collate_fn=collate)

    schedule = tcfg.get("noise_schedule", "linear")
    diff = DiffusionProcess(tcfg["T"], tcfg["beta_start"], tcfg["beta_end"], device,
                            schedule=schedule)
    scdm = SCDM(diff, base_channels=mcfg["base_channels"],
                spatial=mcfg["spatial"], temporal=mcfg["temporal"],
                eeg_hrf=use_hrf)

    ckpt = None
    resume_path = args.resume or tcfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            scdm.load_state_dict(ckpt['model'])
            print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
                  f"loss {ckpt.get('loss', '?')})")
        else:
            scdm.load_state_dict(ckpt)
            print("  Loaded raw state_dict")

    lr = tcfg["learning_rate"]
    opt = torch.optim.AdamW(scdm.parameters(), lr=lr, weight_decay=1e-5)

    if isinstance(ckpt, dict) and 'optimizer' in ckpt:
        try:
            opt.load_state_dict(ckpt['optimizer'])
            for pg in opt.param_groups:
                pg['lr'] = lr
            print("  Restored optimizer state")
        except Exception:
            print("  Optimizer state incompatible, starting fresh optimizer")

    accum = tcfg.get("grad_accum_steps", 1)
    clip = tcfg.get("grad_clip_norm", 1.0)
    ema_decay = tcfg.get("ema_decay", 0.9999)
    warmup = tcfg.get("warmup_epochs", 50)
    epochs = tcfg["epochs"]
    save_every = tcfg.get("save_every", 100)

    trainer = Trainer(scdm, opt, device,
                      grad_accum_steps=accum,
                      grad_clip_norm=clip,
                      ema_decay=ema_decay,
                      warmup_epochs=warmup,
                      total_epochs=epochs)

    if isinstance(ckpt, dict) and 'ema' in ckpt and trainer.ema is not None:
        trainer.ema.load_state_dict(ckpt['ema'])
        print("  Restored EMA weights")

    trainer.fit(loader, epochs, save_path="scdm.pt", save_every=save_every)
    print("Training complete. Best model saved as scdm_best.pt")


if __name__ == "__main__":
    main()
