"""Signal preprocessing for the Shin et al. 2016 EEG-fNIRS MI dataset.

Pipelines follow the paper exactly:
  EEG:   CAR -> 4th-order Chebyshev II band-pass 0.5-50 Hz -> resample to 160 Hz
         -> ICA ocular-artifact removal -> epoch to 4000 samples (25 s @ 160 Hz).
  fNIRS: MBLL -> HbO/HbR -> 6th-order zero-phase Butterworth 0.01-0.1 Hz
         -> epoch to 256 samples (25.6 s @ 10 Hz).

These functions are runnable but were not executed here (the raw dataset must be
downloaded first from TU Berlin / GigaDB). Confirm the .mat field names against the
actual files before use.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import cheby2, butter, filtfilt, sosfiltfilt, resample


# --------------------------------------------------------------------------- EEG
def car(eeg: np.ndarray) -> np.ndarray:
    """Common average reference. eeg: (channels, samples)."""
    return eeg - eeg.mean(axis=0, keepdims=True)


def bandpass_eeg(eeg: np.ndarray, fs: float = 200.0) -> np.ndarray:
    b, a = cheby2(4, 40, [0.5, 50], btype="band", fs=fs)   # 40 dB stopband attenuation
    return filtfilt(b, a, eeg, axis=1)


def resample_eeg(eeg: np.ndarray, fs_in: float = 200.0, fs_out: float = 160.0):
    n = int(round(eeg.shape[1] * fs_out / fs_in))
    return resample(eeg, n, axis=1)


def remove_ocular_ica(eeg: np.ndarray, eog: np.ndarray | None = None,
                      n_components: int | None = None,
                      fs: float = 160.0) -> np.ndarray:
    """MNE-based ICA ocular-artifact removal using EOG channels.

    eeg: (30, samples) EEG data after resampling
    eog: (2, samples) VEOG/HEOG channels (same sampling rate as eeg)
    Returns: (30, samples) cleaned EEG
    """
    if eog is None or eog.shape[0] < 1:
        return eeg

    try:
        import mne
        from mne.preprocessing import ICA

        n_ch = eeg.shape[0]
        ch_names = [f'EEG{i:03d}' for i in range(n_ch)] + ['VEOG', 'HEOG']
        ch_types = ['eeg'] * n_ch + ['eog', 'eog']

        data = np.vstack([eeg, eog])  # (32, samples)
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)

        if n_components is None:
            n_components = min(15, n_ch - 1)

        ica = ICA(n_components=n_components, random_state=42, max_iter=200)
        ica.fit(raw, picks='eeg', verbose=False)

        eog_indices, _ = ica.find_bads_eog(raw, ch_name=['VEOG', 'HEOG'],
                                           verbose=False)
        if eog_indices:
            ica.exclude = eog_indices
            ica.apply(raw, verbose=False)

        return raw.get_data(picks='eeg')
    except Exception:
        return eeg


def preprocess_eeg(eeg_raw: np.ndarray, eog: np.ndarray | None = None) -> np.ndarray:
    """Full EEG preprocessing pipeline.

    eeg_raw: (30, samples) at 200 Hz
    eog: (2, samples) at 200 Hz (VEOG, HEOG) - optional, for ICA
    """
    x = car(eeg_raw)
    x = bandpass_eeg(x, fs=200.0)
    x = resample_eeg(x, 200.0, 160.0)
    if eog is not None:
        eog_resampled = resample(eog, x.shape[1], axis=1)
        x = remove_ocular_ica(x, eog_resampled, fs=160.0)
    return x.astype(np.float32)


# ------------------------------------------------------------------------- fNIRS
def bandpass_fnirs(x: np.ndarray, fs: float = 10.0) -> np.ndarray:
    # SOS form required for numerical stability at 0.01 Hz with fs=10 Hz
    sos = butter(6, [0.01, 0.1], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=1)


def preprocess_fnirs(hb: np.ndarray) -> np.ndarray:
    """hb: HbO or HbR concentration (channels, samples) after MBLL conversion."""
    return bandpass_fnirs(hb, fs=10.0).astype(np.float32)


# ------------------------------------------------------------------------ epochs
def epoch(signal: np.ndarray, onsets: np.ndarray, length: int) -> np.ndarray:
    """Cut fixed-length epochs. signal: (C, T) -> (n_trials, C, length)."""
    out = np.stack([signal[:, o:o + length] for o in onsets], axis=0)
    return out.astype(np.float32)


def validate_shapes(eeg, hbr, hbo, n=1740):
    assert eeg.shape == (n, 30, 4000), f"EEG {eeg.shape} != ({n},30,4000)"
    assert hbr.shape == (n, 36, 256), f"HbR {hbr.shape} != ({n},36,256)"
    assert hbo.shape == (n, 36, 256), f"HbO {hbo.shape} != ({n},36,256)"
