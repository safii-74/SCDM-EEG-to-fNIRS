"""Build training arrays from the Shin et al. 2016 dataset.

Produces:
  eeg.npy      (1740, 30, 4000)  float32
  hbr.npy      (1740, 36, 256)   float32
  hbo.npy      (1740, 36, 256)   float32
  labels.npy   (1740,)           int64
  planes_hbr.pt, planes_hbo.pt  cached correlation planes
  montage.npz                   eeg_coords, fnirs_coords

Usage: PYTHONPATH=. python scripts/build_arrays.py
"""
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_shin2016 import build_eeg_dataset, find_subject_dirs
from src.data.load_nirs import build_nirs_dataset, find_nirs_subject_dirs
from src.data.montage import load_eeg_xy, load_nirs_xy, build_montage_coords
from src.data.dataset import SCDMDataset


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = _PROJECT_ROOT / "DATASET"
EEG_ROOT = DATASET_ROOT
NIRS_ROOT = DATASET_ROOT / "NIRS_01-29"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "preprocessed"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load EEG ---
    print("=" * 60)
    print("Loading EEG data...")
    eeg, eeg_labels, eeg_sizes = build_eeg_dataset(EEG_ROOT, expect_subjects=29)
    print(f"  EEG shape: {eeg.shape}, labels: {eeg_labels.shape}")

    # --- Step 2: Load NIRS ---
    print("\nLoading NIRS data...")
    hbr, hbo, nirs_labels, nirs_sizes = build_nirs_dataset(NIRS_ROOT, expect_subjects=29)
    print(f"  HbR shape: {hbr.shape}, HbO: {hbo.shape}, labels: {nirs_labels.shape}")

    # --- Step 3: Verify trial alignment ---
    print("\nVerifying trial alignment...")
    assert len(eeg_sizes) == len(nirs_sizes) == 29
    offset = 0
    for s_idx, (es, ns) in enumerate(zip(eeg_sizes, nirs_sizes)):
        assert es == ns, f"Subject {s_idx+1}: EEG has {es} trials, NIRS has {ns}"
        el = eeg_labels[offset:offset + es]
        nl = nirs_labels[offset:offset + ns]
        assert np.array_equal(el, nl), \
            f"Subject {s_idx+1}: label mismatch! EEG={el}, NIRS={nl}"
        offset += es
    print(f"  All {offset} trials aligned across EEG and NIRS.")

    # --- Step 4: Z-score per channel within each trial ---
    print("\nApplying per-channel z-scoring...")
    for i in range(eeg.shape[0]):
        mu = eeg[i].mean(axis=1, keepdims=True)
        std = eeg[i].std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        eeg[i] = (eeg[i] - mu) / std

    for i in range(hbr.shape[0]):
        for arr in [hbr, hbo]:
            mu = arr[i].mean(axis=1, keepdims=True)
            std = arr[i].std(axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            arr[i] = (arr[i] - mu) / std

    # --- Step 5: Save arrays ---
    print("\nSaving arrays...")
    np.save(OUTPUT_DIR / "eeg.npy", eeg.astype(np.float32))
    np.save(OUTPUT_DIR / "hbr.npy", hbr.astype(np.float32))
    np.save(OUTPUT_DIR / "hbo.npy", hbo.astype(np.float32))
    np.save(OUTPUT_DIR / "labels.npy", eeg_labels)
    print(f"  eeg.npy: {eeg.shape}")
    print(f"  hbr.npy: {hbr.shape}")
    print(f"  hbo.npy: {hbo.shape}")
    print(f"  labels.npy: {eeg_labels.shape}, balance: "
          f"LMI={np.sum(eeg_labels==0)}, RMI={np.sum(eeg_labels==1)}")

    # --- Step 6: Build montage ---
    print("\nBuilding montage...")
    eeg_dirs = find_subject_dirs(EEG_ROOT)
    nirs_dirs = find_nirs_subject_dirs(NIRS_ROOT)
    eeg_xy = load_eeg_xy(eeg_dirs[0])
    nirs_xy = load_nirs_xy(nirs_dirs[0])
    eeg_coords, fnirs_coords = build_montage_coords(eeg_xy, nirs_xy)
    np.savez(OUTPUT_DIR / "montage.npz",
             eeg_coords=eeg_coords, fnirs_coords=fnirs_coords)
    print(f"  EEG coords: {len(eeg_coords)} channels")
    print(f"  fNIRS coords: {len(fnirs_coords)} channels")

    # Verify no collisions
    all_cells = set()
    for d in [eeg_coords, fnirs_coords]:
        for _, cell in d.items():
            assert cell not in all_cells, f"Collision at {cell}!"
            all_cells.add(cell)
    print(f"  No collisions in 16x16 grid ({len(all_cells)} unique cells)")

    # --- Step 7: Precompute correlation planes ---
    print("\nPrecomputing HbR correlation planes...")
    ds_hbr = SCDMDataset(eeg, hbr, eeg_labels, eeg_coords, fnirs_coords,
                         cache_path=str(OUTPUT_DIR / "planes_hbr.pt"))
    ds_hbr.precompute_planes()

    print("\nPrecomputing HbO correlation planes...")
    ds_hbo = SCDMDataset(eeg, hbo, eeg_labels, eeg_coords, fnirs_coords,
                         cache_path=str(OUTPUT_DIR / "planes_hbo.pt"))
    ds_hbo.precompute_planes()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Total trials: {eeg.shape[0]}")
    print(f"  Label balance: LMI={np.sum(eeg_labels==0)}, RMI={np.sum(eeg_labels==1)}")


if __name__ == "__main__":
    main()
