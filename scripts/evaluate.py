"""Generate synthetic fNIRS and evaluate similarity.

Usage: PYTHONPATH=. python scripts/evaluate.py --config configs/config.yaml --ckpt scdm_best.pt
       PYTHONPATH=. python scripts/evaluate.py --ckpt scdm_best.pt --use-ema --ddim 100
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.correlations import build_coords16
from src.data.dataset import SCDMDataset, collate
from src.data.hrf_features import compute_and_save as compute_hrf
from src.models.scdm import SCDM, DiffusionProcess
from src.evaluation.metrics import signal_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="scdm_best.pt")
    ap.add_argument("--use-ema", action="store_true", help="Use EMA weights for sampling")
    ap.add_argument("--ddim", type=int, default=0,
                    help="DDIM steps (0 = use full DDPM 1000 steps)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dcfg, tcfg, mcfg = cfg["data"], cfg["training"], cfg["model"]
    device = tcfg["device"] if torch.cuda.is_available() else "cpu"

    use_hrf = mcfg.get("eeg_hrf", False)
    if use_hrf:
        hrf_path = dcfg.get("eeg_hrf_path", "data/preprocessed/eeg_hrf.npy")
        if Path(hrf_path).exists():
            eeg = np.load(hrf_path)
        else:
            print("Computing HRF features...")
            eeg = compute_hrf(dcfg["eeg_path"], hrf_path)
    else:
        eeg = np.load(dcfg["eeg_path"], mmap_mode='r')
    fnirs = np.load(dcfg["hbr_path"] if dcfg["modality"] == "hbr" else dcfg["hbo_path"])
    labels = np.load(dcfg["labels_path"])
    montage_path = dcfg.get("montage_path")
    if montage_path and Path(montage_path).exists():
        m = np.load(montage_path, allow_pickle=True)
        ec, fc = m['eeg_coords'].item(), m['fnirs_coords'].item()
    else:
        rng = np.random.default_rng(0)
        ec = build_coords16(rng.standard_normal((30, 2)))
        fc = build_coords16(rng.standard_normal((36, 2)))

    ds = SCDMDataset(eeg, fnirs, labels, ec, fc, cache_path=dcfg["planes_cache"])
    if ds._planes_mmap is None:
        ds.precompute_planes()
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], collate_fn=collate)

    schedule = tcfg.get("noise_schedule", "linear")
    diff = DiffusionProcess(tcfg["T"], tcfg["beta_start"], tcfg["beta_end"], device,
                            schedule=schedule)
    scdm = SCDM(diff, base_channels=mcfg["base_channels"],
                spatial=mcfg["spatial"], temporal=mcfg["temporal"],
                eeg_hrf=use_hrf).to(device)

    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if a.use_ema and 'ema' in ckpt:
            scdm.load_state_dict(ckpt['ema'])
            print("Loaded EMA weights")
        elif 'model' in ckpt:
            scdm.load_state_dict(ckpt['model'])
            print(f"Loaded model (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.4f})")
        else:
            scdm.load_state_dict(ckpt)
    else:
        scdm.load_state_dict(ckpt)
    scdm.eval()

    use_ddim = a.ddim > 0
    sample_fn = (lambda e, p: scdm.sample_ddim(e, p, steps=a.ddim)) if use_ddim else scdm.sample
    method = f"DDIM-{a.ddim}" if use_ddim else "DDPM-1000"
    print(f"Sampling method: {method}")

    reals, synths = [], []
    n_batches = len(loader)
    for i, (e0, f0, planes, _) in enumerate(loader):
        planes = {k: v.to(device) for k, v in planes.items()}
        syn = sample_fn(e0.to(device), planes).cpu().numpy()
        reals.append(f0.numpy()); synths.append(syn)
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{n_batches}")
    real = np.concatenate(reals); synth = np.concatenate(synths)
    metrics = signal_similarity(real, synth)
    print(f"\nResults ({method}):")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  PCC: {metrics['PCC']:.4f}")
    np.save("synthetic_fnirs.npy", synth)
    print("Saved synthetic_fnirs.npy")


if __name__ == "__main__":
    main()
