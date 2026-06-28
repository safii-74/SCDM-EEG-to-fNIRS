"""NIRS loader for the Shin et al. 2016 dataset.

Each subject folder (NIRS_01-29/subject XX/) contains cnt.mat, mrk.mat, mnt.mat directly.
cnt has 6 cells; MI sessions at indices 0, 2, 4.

NIRS data structure:
  - cnt.x: (samples, 72) raw optical density at 2 wavelengths (760nm, 850nm)
  - Channels are 36 source-detector pairs, first 36 columns = lowWL (760nm),
    next 36 columns = highWL (850nm)
  - fs = 10 Hz
  - mrk uses desc codes 1=LMI, 2=RMI

Conversion path: raw OD at 2 wavelengths -> modified Beer-Lambert Law -> HbO/HbR
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt


def _load_mat(path):
    """Load .mat file, trying scipy first then mat73 for v7.3 files."""
    try:
        import scipy.io
        return scipy.io.loadmat(str(path), squeeze_me=True)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(path))


def find_nirs_subject_dirs(root: str | Path) -> list[Path]:
    """Find and sort NIRS subject directories."""
    root = Path(root)
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("subject")]
    dirs.sort(key=lambda p: int(re.search(r'\d+', p.name).group()))
    return dirs


def _mbll(od_low: np.ndarray, od_high: np.ndarray,
          wl_low: float = 760.0, wl_high: float = 850.0,
          dpf_low: float = 6.26, dpf_high: float = 5.57,
          d: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Modified Beer-Lambert Law to convert optical density changes to HbO/HbR.

    Uses extinction coefficients (in cm^-1/(mol/L)) for 760nm and 850nm.
    Returns concentration changes in arbitrary units (relative, not absolute uM).

    od_low, od_high: (channels, samples) optical density at each wavelength.
    """
    # Extinction coefficients (molar, cm^-1 M^-1) from literature
    # HbO at 760nm, HbR at 760nm, HbO at 850nm, HbR at 850nm
    e_hbo_760 = 1486.5865
    e_hbr_760 = 3843.707
    e_hbo_850 = 2526.391
    e_hbr_850 = 1798.643

    # Build extinction matrix E: (2, 2) - rows=wavelengths, cols=[HbO, HbR]
    E = np.array([
        [e_hbo_760 * dpf_low * d, e_hbr_760 * dpf_low * d],
        [e_hbo_850 * dpf_high * d, e_hbr_850 * dpf_high * d]
    ])

    # Convert OD to delta_OD (relative to mean)
    dod_low = od_low - od_low.mean(axis=1, keepdims=True)
    dod_high = od_high - od_high.mean(axis=1, keepdims=True)

    # Solve for [HbO, HbR] at each channel and timepoint
    # E @ [HbO; HbR] = [dOD_low; dOD_high]
    # [HbO; HbR] = E^-1 @ [dOD_low; dOD_high]
    E_inv = np.linalg.inv(E)

    n_ch, n_t = od_low.shape
    hbo = np.zeros((n_ch, n_t), dtype=np.float64)
    hbr = np.zeros((n_ch, n_t), dtype=np.float64)

    for ch in range(n_ch):
        dod = np.vstack([dod_low[ch:ch+1, :], dod_high[ch:ch+1, :]])  # (2, T)
        conc = E_inv @ dod  # (2, T)
        hbo[ch] = conc[0]
        hbr[ch] = conc[1]

    return hbo, hbr


def load_nirs_subject(subject_dir: str | Path, pre_s: float = 5.0,
                      post_s: float = 20.6, fs_proc: float = 10.0
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess fNIRS for one subject.

    Returns:
        hbr: (n_trials, 36, 256) float32
        hbo: (n_trials, 36, 256) float32
        labels: (n_trials,) int64, 0=LMI, 1=RMI
    """
    subject_dir = Path(subject_dir)

    cnt_data = _load_mat(subject_dir / "cnt.mat")
    mrk_data = _load_mat(subject_dir / "mrk.mat")

    cnt = cnt_data['cnt']
    mrk = mrk_data['mrk']

    mi_indices = [0, 2, 4]
    all_hbr, all_hbo, all_labels = [], [], []

    # Butterworth bandpass for fNIRS: 0.01-0.1 Hz, 6th order (SOS for numerical stability)
    sos = butter(6, [0.01, 0.1], btype='band', fs=fs_proc, output='sos')

    epoch_len = 256  # 25.6s at 10 Hz

    for idx in mi_indices:
        if hasattr(cnt, 'shape') and cnt.ndim == 1:
            c = cnt[idx]
        else:
            c = cnt[0, idx] if cnt.ndim == 2 else cnt[idx]

        if hasattr(mrk, 'shape') and mrk.ndim == 1:
            m = mrk[idx]
        else:
            m = mrk[0, idx] if mrk.ndim == 2 else mrk[idx]

        # Extract data
        if hasattr(c, 'dtype') and c.dtype.names:
            x = c['x']
            if hasattr(x, 'shape') and x.ndim == 0:
                x = x.item()
            fs_val = c['fs']
            if hasattr(fs_val, 'shape') and fs_val.ndim == 0:
                fs_val = fs_val.item()
            fs = float(fs_val)
        else:
            x = c['x']
            fs = float(c['fs'])
        x = np.array(x, dtype=np.float64)  # (samples, 72)

        # Split into low wavelength (760nm) and high wavelength (850nm)
        # First 36 columns = lowWL, next 36 = highWL
        n_ch = 36
        od_low = x[:, :n_ch].T    # (36, samples)
        od_high = x[:, n_ch:].T   # (36, samples)

        # Apply MBLL to get HbO/HbR
        hbo_cont, hbr_cont = _mbll(od_low, od_high)

        # Bandpass filter 0.01-0.1 Hz
        hbo_filt = sosfiltfilt(sos, hbo_cont, axis=1).astype(np.float64)
        hbr_filt = sosfiltfilt(sos, hbr_cont, axis=1).astype(np.float64)

        # Extract markers
        if hasattr(m, 'dtype') and m.dtype.names:
            time_val = m['time']
            if hasattr(time_val, 'shape') and time_val.ndim == 0:
                time_val = time_val.item()
            time_ms = np.array(time_val).flatten().astype(np.float64)

            evt = m['event']
            if hasattr(evt, 'shape') and evt.ndim == 0:
                evt = evt.item()
            if hasattr(evt, 'dtype') and evt.dtype.names and 'desc' in evt.dtype.names:
                desc_val = evt['desc']
                if hasattr(desc_val, 'shape') and desc_val.ndim == 0:
                    desc_val = desc_val.item()
                desc = np.array(desc_val).flatten().astype(int)
            else:
                desc = np.array(evt['desc']).flatten().astype(int)
        else:
            time_ms = np.array(m['time']).flatten().astype(np.float64)
            desc = np.array(m['event']['desc']).flatten().astype(int)

        # NIRS uses 1=LMI, 2=RMI -> convert to 0=LMI, 1=RMI
        labels = (desc - 1).astype(np.int64)

        # Resample if fs != fs_proc (already 10 Hz in this dataset)
        if abs(fs - fs_proc) > 0.1:
            from scipy.signal import resample as sig_resample
            n_out = int(round(hbo_filt.shape[1] * fs_proc / fs))
            hbo_filt = sig_resample(hbo_filt, n_out, axis=1)
            hbr_filt = sig_resample(hbr_filt, n_out, axis=1)

        # Epoch
        for trial_idx in range(len(time_ms)):
            onset_sample = int(round(time_ms[trial_idx] / 1000.0 * fs_proc))
            start = onset_sample - int(round(pre_s * fs_proc))

            if start < 0:
                start = 0
            if start + epoch_len > hbo_filt.shape[1]:
                start = hbo_filt.shape[1] - epoch_len

            if start + epoch_len <= hbo_filt.shape[1]:
                all_hbo.append(hbo_filt[:, start:start + epoch_len])
                all_hbr.append(hbr_filt[:, start:start + epoch_len])
                all_labels.append(labels[trial_idx])

    hbr_stacked = np.stack(all_hbr, axis=0)
    hbo_stacked = np.stack(all_hbo, axis=0)
    # Clip extreme values before float32 cast to avoid overflow
    fmax = np.finfo(np.float32).max
    hbr_stacked = np.clip(hbr_stacked, -fmax, fmax)
    hbo_stacked = np.clip(hbo_stacked, -fmax, fmax)
    hbr_out = hbr_stacked.astype(np.float32)
    hbo_out = hbo_stacked.astype(np.float32)
    labels_out = np.array(all_labels, dtype=np.int64)
    return hbr_out, hbo_out, labels_out


def build_nirs_dataset(nirs_root: str | Path, expect_subjects: int = 29):
    """Load all NIRS subjects. Returns (hbr, hbo, labels, sizes)."""
    dirs = find_nirs_subject_dirs(Path(nirs_root))
    assert len(dirs) == expect_subjects, f"Found {len(dirs)} subjects, expected {expect_subjects}"

    all_hbr, all_hbo, all_labels, sizes = [], [], [], []
    for i, d in enumerate(dirs):
        print(f"  Loading NIRS subject {i+1}/{len(dirs)}: {d.name}")
        hbr, hbo, labels = load_nirs_subject(d)
        all_hbr.append(hbr)
        all_hbo.append(hbo)
        all_labels.append(labels)
        sizes.append(len(labels))

    return (np.concatenate(all_hbr, axis=0),
            np.concatenate(all_hbo, axis=0),
            np.concatenate(all_labels, axis=0),
            sizes)
