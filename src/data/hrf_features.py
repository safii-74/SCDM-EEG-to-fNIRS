"""HRF-convolved EEG power envelopes aligned to fNIRS timescale.

Pipeline: raw EEG (30, 4000) @ 160 Hz
  -> band power envelopes (Hilbert transform)
  -> convolve with canonical HRF (double-gamma, peaks ~5s)
  -> downsample to 10 Hz (30, 256)
  -> z-score per channel across all trials
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


def canonical_hrf(fs: float = 160.0, duration: float = 25.0) -> np.ndarray:
    """Double-gamma HRF (Glover 1999). Returns kernel at sampling rate fs."""
    from scipy.special import gamma as gammafn
    t = np.arange(0, duration, 1.0 / fs)
    a1, b1 = 6.0, 1.0
    a2, b2 = 16.0, 1.0
    c = 1.0 / 6.0
    h = ((t ** (a1 - 1) * np.exp(-t / b1)) / (b1 ** a1 * gammafn(a1))
         - c * (t ** (a2 - 1) * np.exp(-t / b2)) / (b2 ** a2 * gammafn(a2)))
    h[t < 0] = 0
    return (h / np.abs(h).sum()).astype(np.float32)


def eeg_to_hrf_features(eeg: np.ndarray, fs_eeg: float = 160.0,
                         n_out: int = 256) -> np.ndarray:
    """Single trial: (30, 4000) -> (30, 256) HRF-convolved power envelope."""
    from scipy.signal import hilbert, fftconvolve, resample
    n_ch, n_samp = eeg.shape
    power = np.abs(hilbert(eeg, axis=1)) ** 2
    hrf = canonical_hrf(fs_eeg)
    out = np.empty((n_ch, n_samp), dtype=np.float32)
    for ch in range(n_ch):
        conv = fftconvolve(power[ch], hrf, mode='full')[:n_samp]
        out[ch] = conv
    return resample(out, n_out, axis=1).astype(np.float32)


def compute_and_save(eeg_path: str, out_path: str,
                     fs_eeg: float = 160.0, n_out: int = 256) -> np.ndarray:
    """Compute HRF features for all trials, z-score, and save.

    Returns: (N, 30, 256) float32 array.
    """
    eeg = np.load(eeg_path, mmap_mode='r')
    N = eeg.shape[0]
    result = np.empty((N, eeg.shape[1], n_out), dtype=np.float32)
    for i in range(N):
        result[i] = eeg_to_hrf_features(np.array(eeg[i]), fs_eeg, n_out)
        if (i + 1) % 100 == 0:
            print(f"  HRF features: {i + 1}/{N}", flush=True)
    print(f"  HRF features: {N}/{N}", flush=True)

    # z-score per channel across all trials (preserves cross-trial amplitude)
    mean = result.mean(axis=(0, 2), keepdims=True)
    std = result.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)
    result = ((result - mean) / std).astype(np.float32)

    np.save(out_path, result)
    print(f"  Saved {out_path}  shape={result.shape}  "
          f"mean={result.mean():.4f}  std={result.std():.4f}")
    return result
