"""Evaluation metrics.

Label convention: LMI = 0 (positive class of interest), RMI = 1.
  ACC  overall accuracy
  PRE  precision for LMI         -> pos_label=0   (corrected; plan used default 1)
  SEN  sensitivity/recall for LMI -> pos_label=0
  SPE  specificity = recall for RMI -> pos_label=1
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "PRE": precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SEN": recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "SPE": recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
    }


def signal_similarity(real: np.ndarray, synth: np.ndarray) -> dict:
    """Per-sample similarity between real and synthetic fNIRS, averaged.
    real/synth: (N, C, L)."""
    real = np.asarray(real, np.float64)
    synth = np.asarray(synth, np.float64)
    mse = np.mean((real - synth) ** 2)
    # Pearson correlation over flattened channel-time per sample
    corrs = []
    for r, s in zip(real, synth):
        rf, sf = r.ravel(), s.ravel()
        rf, sf = rf - rf.mean(), sf - sf.mean()
        d = np.sqrt((rf ** 2).sum() * (sf ** 2).sum()) + 1e-12
        corrs.append(float((rf * sf).sum() / d))
    return {"MSE": float(mse), "PCC": float(np.mean(corrs))}
