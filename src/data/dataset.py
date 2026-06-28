"""SCDM dataset.

Returns per sample: e0 (30,4000), f0 (36,256), the four 16x16 correlation planes,
and the label. Correlation planes are EXPENSIVE (distance correlation) so they are
computed once and cached to disk as mmap-friendly numpy arrays, never inside the
model forward pass.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .correlations import make_correlation_planes

_PLANE_KEYS = ("cef", "cfe", "ce", "cf")
_PLANE_CHANNELS = {"cef": 30, "cfe": 36, "ce": 30, "cf": 36}


def _npy_paths(cache_path: str) -> dict[str, Path]:
    """Derive per-key .npy paths from the base cache path."""
    base = Path(cache_path).with_suffix("")
    return {k: base.parent / f"{base.name}_{k}.npy" for k in _PLANE_KEYS}


def _migrate_pt_to_npy(pt_path: str, npy_paths: dict[str, Path], N: int):
    """One-time conversion from old .pt cache to 4 mmap-friendly .npy files."""
    print(f"Migrating {pt_path} -> numpy mmap format...")
    old = torch.load(pt_path, weights_only=False)
    for k in _PLANE_KEYS:
        arr = np.stack([old[i][k].numpy() for i in range(N)])
        np.save(str(npy_paths[k]), arr)
    print("  Migration complete. Old .pt file kept as backup.")


class SCDMDataset(Dataset):
    def __init__(self, eeg, fnirs, labels, eeg_coords, fnirs_coords,
                 cache_path: str | None = None):
        """eeg: (N,30,4000)  fnirs: (N,36,256)  labels: (N,)
        *_coords: {channel_idx: (row,col)} 16x16 mappings from build_coords16.
        """
        self.eeg = eeg if (isinstance(eeg, np.ndarray) and eeg.dtype == np.float32) else np.asarray(eeg, dtype=np.float32)
        self.fnirs = np.asarray(fnirs, dtype=np.float32)
        self.labels = np.asarray(labels).astype(np.int64)
        self.eeg_coords, self.fnirs_coords = eeg_coords, fnirs_coords
        self.cache_path = cache_path
        self._planes_mmap = None

        if cache_path:
            npys = _npy_paths(cache_path)
            all_npy_exist = all(p.exists() for p in npys.values())

            if not all_npy_exist and os.path.exists(cache_path):
                _migrate_pt_to_npy(cache_path, npys, len(self.labels))
                all_npy_exist = True

            if all_npy_exist:
                self._planes_mmap = {
                    k: np.load(str(npys[k]), mmap_mode='r') for k in _PLANE_KEYS
                }

    def precompute_planes(self):
        """Build and cache correlation planes for all samples (run once)."""
        N = len(self)
        arrays = {k: np.zeros((N, _PLANE_CHANNELS[k], 16, 16), dtype=np.float32)
                  for k in _PLANE_KEYS}
        for i in range(N):
            p = make_correlation_planes(
                self.eeg[i], self.fnirs[i], self.eeg_coords, self.fnirs_coords)
            for k in _PLANE_KEYS:
                arrays[k][i] = p[k].numpy()
            if (i + 1) % 50 == 0:
                print(f"  planes {i + 1}/{N}")
        if self.cache_path:
            npys = _npy_paths(self.cache_path)
            for k in _PLANE_KEYS:
                np.save(str(npys[k]), arrays[k])
        self._planes_mmap = {
            k: np.load(str(_npy_paths(self.cache_path)[k]), mmap_mode='r')
            for k in _PLANE_KEYS
        } if self.cache_path else arrays
        return self

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self._planes_mmap is None:
            planes = make_correlation_planes(
                self.eeg[i], self.fnirs[i], self.eeg_coords, self.fnirs_coords)
        else:
            planes = {k: torch.from_numpy(self._planes_mmap[k][i].copy())
                      for k in _PLANE_KEYS}
        return {
            "e0": torch.from_numpy(self.eeg[i].copy()),
            "f0": torch.from_numpy(self.fnirs[i].copy()),
            "planes": planes,
            "label": int(self.labels[i]),
        }


def collate(batch):
    e0 = torch.stack([b["e0"] for b in batch])
    f0 = torch.stack([b["f0"] for b in batch])
    keys = ("cef", "cfe", "ce", "cf")
    planes = {k: torch.stack([b["planes"][k] for b in batch]) for k in keys}
    labels = torch.tensor([b["label"] for b in batch])
    return e0, f0, planes, labels
