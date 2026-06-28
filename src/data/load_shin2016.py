"""EEG loader for the Shin et al. 2016 dataset (BBCI format).

Each subject folder contains cnt.mat, mrk.mat, mnt.mat inside a 'with occular artifact'
subfolder. cnt has 6 cells; MI sessions are at indices 0, 2, 4. Each MI session has
~20 trials (10 LMI + 10 RMI) => 60 trials per subject.

EEG markers use desc codes 16=LMI, 32=RMI.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
from scipy.signal import resample

from .preprocessing import preprocess_eeg


def _load_mat(path):
    """Load .mat file, trying scipy first then mat73 for v7.3 files."""
    try:
        import scipy.io
        return scipy.io.loadmat(str(path), squeeze_me=True)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(path))


def find_subject_dirs(root: str | Path, prefix: str = "subject") -> list[Path]:
    """Find and sort EEG subject directories across EEG_* grouped folders."""
    root = Path(root)
    dirs = []
    for group in sorted(root.iterdir()):
        if not group.is_dir() or not group.name.startswith('EEG_'):
            continue
        for subj in sorted(group.iterdir()):
            if subj.is_dir() and subj.name.startswith(prefix):
                dirs.append(subj)
    dirs.sort(key=lambda p: int(re.search(r'\d+', p.name).group()))
    return dirs


def _get_eeg_data_dir(subject_dir: Path) -> Path:
    """Get the path to the 'with occular artifact' subfolder."""
    d = subject_dir / "with occular artifact"
    if d.exists():
        return d
    return subject_dir


def load_eeg_subject(subject_dir: str | Path, pre_s: float = 5.0, post_s: float = 20.0,
                     fs_proc: float = 160.0) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess EEG for one subject.

    Returns:
        eeg: (n_trials, 30, 4000) float32 at 160 Hz
        labels: (n_trials,) int64, 0=LMI, 1=RMI
    """
    subject_dir = Path(subject_dir)
    data_dir = _get_eeg_data_dir(subject_dir)

    cnt_data = _load_mat(data_dir / "cnt.mat")
    mrk_data = _load_mat(data_dir / "mrk.mat")

    cnt = cnt_data['cnt']
    mrk = mrk_data['mrk']

    mi_indices = [0, 2, 4]
    all_epochs = []
    all_labels = []

    for idx in mi_indices:
        if hasattr(cnt, 'shape') and cnt.ndim == 1:
            c = cnt[idx]
        else:
            c = cnt[0, idx] if cnt.ndim == 2 else cnt[idx]

        if hasattr(mrk, 'shape') and mrk.ndim == 1:
            m = mrk[idx]
        else:
            m = mrk[0, idx] if mrk.ndim == 2 else mrk[idx]

        # Extract continuous data
        if hasattr(c, 'dtype') and c.dtype.names:
            x = c['x']
            if hasattr(x, 'shape') and x.ndim == 0:
                x = x.item()
            fs_val = c['fs']
            if hasattr(fs_val, 'shape') and fs_val.ndim == 0:
                fs_val = fs_val.item()
            fs = float(np.array(fs_val).flat[0])
            clab_raw = c['clab']
            if hasattr(clab_raw, 'shape') and clab_raw.ndim == 0:
                clab_raw = clab_raw.item()
        else:
            x = c['x']
            fs = float(c['fs'])
            clab_raw = c['clab']

        x = np.array(x, dtype=np.float64)  # (samples, channels)

        # Get channel labels
        if hasattr(clab_raw, 'flatten'):
            clab_raw = clab_raw.flatten()
        clab = [str(cl[0]) if hasattr(cl, '__len__') and not isinstance(cl, str) else str(cl)
                for cl in clab_raw]

        # Find EEG channels (first 30, excluding VEOG/HEOG)
        eeg_idx = [i for i, name in enumerate(clab) if name not in ('VEOG', 'HEOG')][:30]
        eog_idx = [i for i, name in enumerate(clab) if name in ('VEOG', 'HEOG')]

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

        time_ms = np.array(time_ms).flatten().astype(np.float64)
        desc = np.array(desc).flatten().astype(int)

        # EEG uses 16=LMI, 32=RMI
        labels = np.where(desc == 16, 0, 1).astype(np.int64)

        # Preprocess: CAR, bandpass, resample, ICA
        eeg_full = x[:, eeg_idx].T  # (30, samples)
        eog_full = x[:, eog_idx].T if eog_idx else None  # (2, samples) for ICA

        eeg_proc = preprocess_eeg(eeg_full, eog=eog_full)  # (30, samples_at_160Hz)

        # Epoch: 25s window (5s pre + 20s post) at 160 Hz = 4000 samples
        epoch_len = int(round((pre_s + post_s) * fs_proc))  # 4000
        for trial_idx in range(len(time_ms)):
            onset_sample_orig = int(round(time_ms[trial_idx] / 1000.0 * fs))
            onset_at_proc = int(round(time_ms[trial_idx] / 1000.0 * fs_proc))
            start = onset_at_proc - int(round(pre_s * fs_proc))

            if start < 0:
                start = 0
            if start + epoch_len > eeg_proc.shape[1]:
                start = eeg_proc.shape[1] - epoch_len

            trial = eeg_proc[:, start:start + epoch_len]
            if trial.shape[1] == epoch_len:
                all_epochs.append(trial)
                all_labels.append(labels[trial_idx])

    eeg_out = np.stack(all_epochs, axis=0).astype(np.float32)
    labels_out = np.array(all_labels, dtype=np.int64)
    return eeg_out, labels_out


def build_eeg_dataset(eeg_root: str | Path, expect_subjects: int = 29):
    """Load all EEG subjects. Returns (eeg, labels, sizes).

    eeg: (N, 30, 4000), labels: (N,), sizes: list of per-subject trial counts.
    """
    dirs = find_subject_dirs(eeg_root)
    assert len(dirs) == expect_subjects, f"Found {len(dirs)} subjects, expected {expect_subjects}"

    all_eeg, all_labels, sizes = [], [], []
    for i, d in enumerate(dirs):
        print(f"  Loading EEG subject {i+1}/{len(dirs)}: {d.name}")
        eeg, labels = load_eeg_subject(d)
        all_eeg.append(eeg)
        all_labels.append(labels)
        sizes.append(len(labels))

    return (np.concatenate(all_eeg, axis=0),
            np.concatenate(all_labels, axis=0),
            sizes)
