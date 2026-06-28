"""Run all six ablation variants and produce classification results.

Trains each (spatial, temporal) combination, generates synthetic fNIRS,
classifies EEG+synthetic, and emits a table matching the paper's Table I.

Usage: PYTHONPATH=. python scripts/run_ablations.py
"""
from pathlib import Path
import numpy as np
import torch
import yaml
import csv
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SCDMDataset, collate
from src.models.scdm import SCDM, DiffusionProcess
from src.training.trainer import Trainer
from src.evaluation.classifier import run_evaluation


OUTPUT_DIR = Path(r"E:\SCDM — EEG-to-fNIRS Cross-Modal Generation\scdm\data\preprocessed")
RESULTS_DIR = Path(r"E:\SCDM — EEG-to-fNIRS Cross-Modal Generation\scdm\results")


ABLATION_CONFIGS = [
    ("attn", "cov"),
    ("attn", "mtr"),
    ("scg_eeg", "cov"),
    ("scg_eeg", "mtr"),
    ("scg_fnirs", "cov"),
    ("scg_fnirs", "mtr"),
]


def load_data(modality="hbr"):
    """Load preprocessed arrays and montage."""
    eeg = np.load(OUTPUT_DIR / "eeg.npy", mmap_mode='r')
    fnirs = np.load(OUTPUT_DIR / f"{modality}.npy")
    labels = np.load(OUTPUT_DIR / "labels.npy")
    montage = np.load(OUTPUT_DIR / "montage.npz", allow_pickle=True)
    eeg_coords = montage['eeg_coords'].item()
    fnirs_coords = montage['fnirs_coords'].item()
    return eeg, fnirs, labels, eeg_coords, fnirs_coords


def train_variant(spatial, temporal, eeg, fnirs, labels, eeg_coords, fnirs_coords,
                  modality="hbr", epochs=100, batch_size=4,
                  grad_accum_steps=4, device="cpu"):
    """Train one ablation variant and return synthetic fNIRS."""
    planes_cache = str(OUTPUT_DIR / f"planes_{modality}.pt")
    ds = SCDMDataset(eeg, fnirs, labels, eeg_coords, fnirs_coords,
                     cache_path=planes_cache)
    if ds._planes_mmap is None:
        ds.precompute_planes()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    diff = DiffusionProcess(T=1000, device=device)
    scdm = SCDM(diff, base_channels=32, spatial=spatial, temporal=temporal)
    opt = torch.optim.Adam(scdm.parameters(), lr=1e-4)
    trainer = Trainer(scdm, opt, device, grad_accum_steps=grad_accum_steps)
    trainer.fit(loader, epochs)

    # Generate synthetic
    scdm.eval()
    synths = []
    eval_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for e0, f0, planes, _ in eval_loader:
            planes_dev = {k: v.to(device) for k, v in planes.items()}
            syn = scdm.sample(e0.to(device), planes_dev).cpu().numpy()
            synths.append(syn)
    return np.concatenate(synths, axis=0)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for modality in ["hbr", "hbo"]:
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY - {modality.upper()}")
        print(f"{'='*60}")

        eeg, fnirs, labels, eeg_coords, fnirs_coords = load_data(modality)
        results_rows = []

        for spatial, temporal in ABLATION_CONFIGS:
            variant_name = f"{spatial}+{temporal}"
            print(f"\n--- Training {variant_name} ---")

            synth = train_variant(spatial, temporal, eeg, fnirs, labels,
                                  eeg_coords, fnirs_coords, modality,
                                  epochs=100, device=device)

            # Classify EEG+synthetic
            print(f"  Evaluating {variant_name}...")
            eval_results = run_evaluation(
                eeg, labels, fnirs_real=fnirs, fnirs_synth=synth,
                train_ratios=[0.5], n_runs=3, epochs=30, device=device
            )

            # Extract 5:5 split results
            split_key = "5:5"
            row = {"variant": variant_name, "modality": modality}
            if split_key in eval_results:
                for mode in ["eeg_only", "eeg_real", "eeg_synth"]:
                    if mode in eval_results[split_key] and eval_results[split_key][mode]:
                        for metric, val in eval_results[split_key][mode].items():
                            row[f"{mode}_{metric}"] = val
            results_rows.append(row)
            print(f"  {variant_name}: {row}")

        # Save results to CSV
        csv_path = RESULTS_DIR / f"ablation_{modality}.csv"
        if results_rows:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
                writer.writeheader()
                writer.writerows(results_rows)
            print(f"\nResults saved to {csv_path}")

    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")


if __name__ == "__main__":
    main()
