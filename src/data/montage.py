"""Montage / coordinate mapping for the 16x16 spatial plane.

The 16x16 plane uses a SINGLE shared grid over all 66 channels (30 EEG + 36 fNIRS),
not two independent binnings, so spatial relationships are comparable across modalities.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io

from .correlations import build_coords16


def _load_mat(path):
    try:
        return scipy.io.loadmat(str(path), squeeze_me=True)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(path))


def load_eeg_xy(eeg_subject_dir: str | Path) -> np.ndarray:
    """Load 2D EEG electrode positions (30 channels, excluding VEOG/HEOG).

    Returns: (30, 2) array of (x, y) scalp coordinates.
    """
    eeg_dir = Path(eeg_subject_dir)
    data_dir = eeg_dir / "with occular artifact"
    if not data_dir.exists():
        data_dir = eeg_dir

    mnt_data = _load_mat(data_dir / "mnt.mat")
    mnt = mnt_data['mnt']

    if hasattr(mnt, 'dtype') and mnt.dtype.names:
        x = mnt['x'].item() if hasattr(mnt['x'], 'item') else mnt['x']
        y = mnt['y'].item() if hasattr(mnt['y'], 'item') else mnt['y']
    else:
        x = np.array(mnt['x'])
        y = np.array(mnt['y'])

    x = np.array(x).flatten()
    y = np.array(y).flatten()

    # First 30 channels are EEG (last 2 are VEOG, HEOG)
    return np.column_stack([x[:30], y[:30]])


def load_nirs_xy(nirs_subject_dir: str | Path) -> np.ndarray:
    """Load 2D fNIRS channel positions (36 source-detector midpoints).

    Returns: (36, 2) array of (x, y) scalp coordinates.
    """
    nirs_dir = Path(nirs_subject_dir)
    mnt_data = _load_mat(nirs_dir / "mnt.mat")
    mnt = mnt_data['mnt']

    if hasattr(mnt, 'dtype') and mnt.dtype.names:
        x = mnt['x'].item() if hasattr(mnt['x'], 'item') else mnt['x']
        y = mnt['y'].item() if hasattr(mnt['y'], 'item') else mnt['y']
    else:
        x = np.array(mnt['x'])
        y = np.array(mnt['y'])

    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.column_stack([x, y])


def build_montage_coords(eeg_xy: np.ndarray, fnirs_xy: np.ndarray
                         ) -> tuple[dict, dict]:
    """Build 16x16 coordinate mapping using a shared normalization over all 66 channels.

    Both modalities share one normalization so spatial relationships are preserved.

    Returns: (eeg_coords, fnirs_coords) - dicts mapping channel_idx -> (row, col).
    """
    # Stack all 66 channels for shared normalization
    all_pos = np.vstack([eeg_xy, fnirs_xy])  # (66, 2)

    # Normalize to [0, 1] using shared min/max
    mn = all_pos.min(axis=0)
    mx = all_pos.max(axis=0)
    span = np.where((mx - mn) == 0, 1.0, (mx - mn))
    norm = (all_pos - mn) / span  # (66, 2) in [0, 1]

    # Bin to 16x16
    cells = np.clip((norm * 15).round().astype(int), 0, 15)

    # Resolve collisions using spiral search (process in order: EEG first, then fNIRS)
    used = set()
    all_mapping = {}
    for idx in range(66):
        r, c = int(cells[idx, 0]), int(cells[idx, 1])
        if (r, c) in used:
            r, c = _nearest_free(r, c, used)
        used.add((r, c))
        all_mapping[idx] = (r, c)

    # Split into EEG (0-29) and fNIRS (30-65) coordinate dicts
    eeg_coords = {i: all_mapping[i] for i in range(30)}
    fnirs_coords = {i: all_mapping[30 + i] for i in range(36)}

    return eeg_coords, fnirs_coords


def _nearest_free(r, c, used):
    """Spiral search for nearest free cell in 16x16 grid."""
    for radius in range(1, 16):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 16 and 0 <= nc < 16 and (nr, nc) not in used:
                    return nr, nc
    raise RuntimeError("16x16 grid is full")


def load_montage(eeg_subject_dir: str | Path, nirs_subject_dir: str | Path
                 ) -> tuple[dict, dict]:
    """Convenience: load positions and build the shared 16x16 montage."""
    eeg_xy = load_eeg_xy(eeg_subject_dir)
    nirs_xy = load_nirs_xy(nirs_subject_dir)
    return build_montage_coords(eeg_xy, nirs_xy)
