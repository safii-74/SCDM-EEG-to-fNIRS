"""Visualization utilities for SCDM results.

Produces:
  - Hemodynamic response curves (Fig. 4): mean epoch real vs synthetic, HbR/HbO x LMI/RMI
  - Scalp topography (Fig. 5): motor-area channels, 7 windows, real vs synthetic
  - Spatial correspondence (Fig. 3): most-correlated EEG channel per fNIRS channel
  - Similarity metrics: MSE, PCC tables
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from .metrics import signal_similarity


FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "figures"


def plot_hemodynamic_curves(real_hbr, real_hbo, synth_hbr, synth_hbo,
                            labels, fs=10.0, save=True):
    """Plot mean hemodynamic response curves: real vs synthetic for HbR/HbO under LMI/RMI.

    real_hbr, synth_hbr: (N, 36, 256)
    labels: (N,) 0=LMI, 1=RMI
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    time = np.arange(256) / fs - 5.0  # relative to onset (5s pre-stimulus)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    configs = [
        (real_hbr, synth_hbr, "HbR", 0, "LMI"),
        (real_hbr, synth_hbr, "HbR", 1, "RMI"),
        (real_hbo, synth_hbo, "HbO", 0, "LMI"),
        (real_hbo, synth_hbo, "HbO", 1, "RMI"),
    ]

    for idx, (real, synth, hb_type, label_val, label_name) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        mask = labels == label_val

        # Mean across trials and channels
        real_mean = real[mask].mean(axis=(0, 1))
        synth_mean = synth[mask].mean(axis=(0, 1))
        real_std = real[mask].mean(axis=1).std(axis=0)
        synth_std = synth[mask].mean(axis=1).std(axis=0)

        ax.plot(time, real_mean, 'b-', label='Real', linewidth=1.5)
        ax.fill_between(time, real_mean - real_std, real_mean + real_std,
                        alpha=0.2, color='blue')
        ax.plot(time, synth_mean, 'r--', label='Synthetic', linewidth=1.5)
        ax.fill_between(time, synth_mean - synth_std, synth_mean + synth_std,
                        alpha=0.2, color='red')
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{hb_type} (a.u.)')
        ax.set_title(f'{hb_type} - {label_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Hemodynamic Response Curves: Real vs Synthetic', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "hemodynamic_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved hemodynamic_curves.png")


def plot_scalp_topography(real_fnirs, synth_fnirs, fnirs_coords, labels,
                          fs=10.0, save=True):
    """Plot scalp topography in 7 time windows (3-17s), real vs synthetic.

    real_fnirs, synth_fnirs: (N, 36, 256)
    fnirs_coords: {ch_idx: (row, col)} in 16x16 grid
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Time windows: 3-5, 5-7, 7-9, 9-11, 11-13, 13-15, 15-17 seconds
    # With 5s pre-stimulus, these correspond to samples 80-120, 100-120, etc.
    pre_s = 5.0
    windows = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15), (15, 17)]

    # Use LMI trials only for topography
    mask = labels == 0
    real_lmi = real_fnirs[mask].mean(axis=0)  # (36, 256)
    synth_lmi = synth_fnirs[mask].mean(axis=0)

    fig, axes = plt.subplots(2, 7, figsize=(21, 6))

    for w_idx, (t_start, t_end) in enumerate(windows):
        s_start = int((t_start + pre_s) * fs)
        s_end = int((t_end + pre_s) * fs)

        real_val = real_lmi[:, s_start:s_end].mean(axis=1)
        synth_val = synth_lmi[:, s_start:s_end].mean(axis=1)

        for row, (vals, title) in enumerate([(real_val, "Real"), (synth_val, "Synth")]):
            ax = axes[row, w_idx]
            grid = np.full((16, 16), np.nan)
            for ch, (r, c) in fnirs_coords.items():
                grid[r, c] = vals[ch]
            im = ax.imshow(grid, cmap='RdBu_r', interpolation='nearest',
                           vmin=-np.nanmax(np.abs(grid)), vmax=np.nanmax(np.abs(grid)))
            ax.set_title(f"{title}\n{t_start}-{t_end}s", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Scalp Topography (LMI): Real vs Synthetic', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "scalp_topography.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scalp_topography.png")


def plot_spatial_correspondence(eeg, real_fnirs, synth_fnirs, eeg_coords, fnirs_coords,
                                save=True):
    """For each fNIRS channel, find the most-correlated EEG channel (distance correlation).

    Plots real vs synthetic correspondence as a heatmap.
    """
    from src.data.correlations import distance_correlation_matrix

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Compute on mean across trials
    eeg_mean = eeg.mean(axis=0)  # (30, 4000)
    real_mean = real_fnirs.mean(axis=0)  # (36, 256)
    synth_mean = synth_fnirs.mean(axis=0)

    # Distance correlation matrices
    dc_real = distance_correlation_matrix(eeg_mean, real_mean)  # (30, 36)
    dc_synth = distance_correlation_matrix(eeg_mean, synth_mean)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, dc, title in [(axes[0], dc_real, "Real fNIRS"),
                           (axes[1], dc_synth, "Synthetic fNIRS")]:
        sns.heatmap(dc, ax=ax, cmap='viridis', vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False)
        ax.set_xlabel('fNIRS channels (36)')
        ax.set_ylabel('EEG channels (30)')
        ax.set_title(f'Distance Correlation - {title}')

    plt.suptitle('Spatial Correspondence: EEG-fNIRS', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "spatial_correspondence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial_correspondence.png")


def plot_similarity_summary(real_fnirs, synth_fnirs, save=True):
    """Bar chart of MSE and PCC between real and synthetic."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = signal_similarity(real_fnirs, synth_fnirs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(['MSE'], [metrics['MSE']], color='steelblue')
    axes[0].set_title('Mean Squared Error')
    axes[0].set_ylabel('MSE')

    axes[1].bar(['PCC'], [metrics['PCC']], color='darkorange')
    axes[1].set_title('Pearson Correlation Coefficient')
    axes[1].set_ylabel('PCC')
    axes[1].set_ylim([-1, 1])

    plt.suptitle('Signal Similarity: Real vs Synthetic fNIRS', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "similarity_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity_metrics.png")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  PCC: {metrics['PCC']:.6f}")
    return metrics


def generate_all_figures(eeg, real_hbr, real_hbo, synth_hbr, synth_hbo,
                         labels, eeg_coords, fnirs_coords):
    """Generate all visualization figures."""
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    plot_hemodynamic_curves(real_hbr, real_hbo, synth_hbr, synth_hbo, labels)
    plot_scalp_topography(real_hbr, synth_hbr, fnirs_coords, labels)
    plot_spatial_correspondence(eeg, real_hbr, synth_hbr, eeg_coords, fnirs_coords)
    plot_similarity_summary(real_hbr, synth_hbr)

    print("\nAll figures saved to:", FIGURES_DIR)
