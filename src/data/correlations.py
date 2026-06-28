"""Correlation utilities for the SCG module.

Per the paper:
  * Cef, Cfe (cross-modal) use *distance* correlation, because EEG (len 4000) and
    fNIRS (len 256) have different lengths and Pearson cannot be applied directly.
  * Ce, Cf (single-modality, used by SCG-EEG / SCG-fNIRS) use Pearson correlation.
  * Each (Ca, Cb) matrix is projected onto a 16x16 plane built from the 66-channel
    scalp layout (Fig. 3). These planes are PRECOMPUTED PER SAMPLE and cached;
    they are never recomputed inside the U-Net forward pass.
"""
from __future__ import annotations
import numpy as np
import torch

try:
    import dcor
    _HAS_DCOR = True
except Exception:  # pragma: no cover
    _HAS_DCOR = False


def distance_correlation_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Distance correlation between every channel of X and every channel of Y.

    X: (Cx, Lx), Y: (Cy, Ly).  Distance correlation (like Pearson) needs EQUAL-length
    paired samples, so when Lx != Ly the longer series is resampled to the shorter
    length first.  Distance correlation is preferred over Pearson because it also
    captures non-linear dependence -- not because it tolerates unequal lengths
    (it does not).  Returns (Cx, Cy) in [0, 1].
    """
    assert _HAS_DCOR, "dcor is required for distance correlation"
    from scipy.signal import resample
    Lx, Ly = X.shape[1], Y.shape[1]
    if Lx != Ly:
        L = min(Lx, Ly)
        if Lx != L:
            X = resample(X, L, axis=1)
        if Ly != L:
            Y = resample(Y, L, axis=1)
    Cx, Cy = X.shape[0], Y.shape[0]
    M = np.zeros((Cx, Cy), dtype=np.float32)
    for i in range(Cx):
        xi = np.ascontiguousarray(X[i], dtype=np.float64)
        for j in range(Cy):
            M[i, j] = dcor.distance_correlation(xi, np.ascontiguousarray(Y[j], np.float64))
    return M


def pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation among channels of X: (C, L) -> (C, C)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = Xc @ Xc.T
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return (cov / (std[:, None] * std[None, :])).astype(np.float32)


def build_coords16(positions_2d: np.ndarray) -> dict:
    """Map N channels with 2D scalp coordinates to a 16x16 grid.

    positions_2d: (N, 2) array of (x, y) scalp coordinates (any consistent units).
    Returns {channel_index: (row, col)} with row, col in [0, 15].

    NOTE: This is a generic binning of normalized coordinates. For exact paper
    reproduction, replace `positions_2d` with the dataset montage (10-5 electrode
    positions + fNIRS source/detector midpoints) so the 66-channel layout of Fig. 3
    is honoured. Collisions are resolved by nudging to the nearest free cell.
    """
    pos = positions_2d.astype(np.float64)
    mn, mx = pos.min(0), pos.max(0)
    span = np.where((mx - mn) == 0, 1.0, (mx - mn))
    norm = (pos - mn) / span                      # -> [0, 1]
    cells = np.clip((norm * 15).round().astype(int), 0, 15)
    used, mapping = set(), {}
    for idx, (r, c) in enumerate(cells):
        if (r, c) in used:                        # resolve collision: spiral search
            r, c = _nearest_free(r, c, used)
        used.add((r, c))
        mapping[idx] = (int(r), int(c))
    return mapping


def _nearest_free(r, c, used):
    for radius in range(1, 16):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 16 and 0 <= nc < 16 and (nr, nc) not in used:
                    return nr, nc
    raise RuntimeError("16x16 grid is full")


def project_to_16x16(corr: np.ndarray, coords16: dict) -> np.ndarray:
    """corr: (Ca, Cb) -> planes (Ca, 16, 16); cell (r,c) holds corr[:, b]."""
    Ca, Cb = corr.shape
    out = np.zeros((Ca, 16, 16), dtype=np.float32)
    for b in range(Cb):
        r, c = coords16[b]
        out[:, r, c] = corr[:, b]
    return out


def make_correlation_planes(eeg: np.ndarray, fnirs: np.ndarray,
                            eeg_coords: dict, fnirs_coords: dict) -> dict:
    """Build all four projected correlation planes for ONE sample.

    eeg: (30, 4000), fnirs: (36, 256).  Returns torch tensors:
      cef (30,16,16), cfe (36,16,16)  -- cross-modal (distance corr)
      ce  (30,16,16), cf  (36,16,16)  -- single-modality (Pearson)
    Cell layout: Cef/Ce keyed by fNIRS/EEG columns -> fnirs_coords/eeg_coords.
    """
    Cef = distance_correlation_matrix(eeg, fnirs)        # (30, 36)
    Ce = pearson_matrix(eeg)                             # (30, 30)
    Cf = pearson_matrix(fnirs)                           # (36, 36)
    return {
        "cef": torch.from_numpy(project_to_16x16(Cef, fnirs_coords)),       # (30,16,16)
        "cfe": torch.from_numpy(project_to_16x16(Cef.T, eeg_coords)),       # (36,16,16)
        "ce":  torch.from_numpy(project_to_16x16(Ce, eeg_coords)),          # (30,16,16)
        "cf":  torch.from_numpy(project_to_16x16(Cf, fnirs_coords)),        # (36,16,16)
    }
