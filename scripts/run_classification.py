"""Classification evaluation: EEG-only vs EEG+Real vs EEG+Synth fNIRS.

Usage: PYTHONPATH=. python scripts/run_classification.py
"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from src.evaluation.classifier import (
    EEGNet, HybridNet, train_classifier, evaluate_classifier,
)

DS = 8
EPOCHS = 50
N_RUNS = 5
BATCH = 32
LR = 1e-3
DEVICE = "cpu"

print("Loading data...", flush=True)
eeg = np.load("data/preprocessed/eeg.npy")
labels = np.load("data/preprocessed/labels.npy")
hbr = np.load("data/preprocessed/hbr.npy")
synth = np.load("synthetic_fnirs.npy")
print(f"EEG {eeg.shape}, fNIRS {hbr.shape}, Synth {synth.shape}, Labels {labels.shape}", flush=True)
print(f"Config: downsample={DS}, epochs={EPOCHS}, runs={N_RUNS}, lr={LR}", flush=True)

results = {"eeg_only": [], "eeg_real": [], "eeg_synth": []}

for run in range(N_RUNS):
    print(f"\n--- Run {run+1}/{N_RUNS} ---", flush=True)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=run * 42)
    train_idx, test_idx = next(sss.split(eeg, labels))

    eeg_tr = torch.from_numpy(eeg[train_idx])
    eeg_te = torch.from_numpy(eeg[test_idx])
    lab_tr = torch.from_numpy(labels[train_idx])
    lab_te = torch.from_numpy(labels[test_idx])

    # EEG-only
    model = EEGNet(downsample=DS)
    dl_tr = DataLoader(TensorDataset(eeg_tr, lab_tr), batch_size=BATCH, shuffle=True)
    dl_te = DataLoader(TensorDataset(eeg_te, lab_te), batch_size=BATCH)
    model = train_classifier(model, dl_tr, epochs=EPOCHS, lr=LR, device=DEVICE)
    m = evaluate_classifier(model, dl_te, DEVICE)
    results["eeg_only"].append(m)
    print(f"  EEG-only:  ACC={m['ACC']:.1f}%", flush=True)

    # EEG + Real fNIRS
    fnirs_tr = torch.from_numpy(hbr[train_idx])
    fnirs_te = torch.from_numpy(hbr[test_idx])
    model_h = HybridNet(downsample=DS)
    dl_tr_h = DataLoader(TensorDataset(eeg_tr, fnirs_tr, lab_tr), batch_size=BATCH, shuffle=True)
    dl_te_h = DataLoader(TensorDataset(eeg_te, fnirs_te, lab_te), batch_size=BATCH)
    model_h = train_classifier(model_h, dl_tr_h, epochs=EPOCHS, lr=LR, device=DEVICE)
    m = evaluate_classifier(model_h, dl_te_h, DEVICE)
    results["eeg_real"].append(m)
    print(f"  EEG+Real:  ACC={m['ACC']:.1f}%", flush=True)

    # EEG + Synthetic fNIRS
    synth_tr = torch.from_numpy(synth[train_idx])
    synth_te = torch.from_numpy(synth[test_idx])
    model_s = HybridNet(downsample=DS)
    dl_tr_s = DataLoader(TensorDataset(eeg_tr, synth_tr, lab_tr), batch_size=BATCH, shuffle=True)
    dl_te_s = DataLoader(TensorDataset(eeg_te, synth_te, lab_te), batch_size=BATCH)
    model_s = train_classifier(model_s, dl_tr_s, epochs=EPOCHS, lr=LR, device=DEVICE)
    m = evaluate_classifier(model_s, dl_te_s, DEVICE)
    results["eeg_synth"].append(m)
    print(f"  EEG+Synth: ACC={m['ACC']:.1f}%", flush=True)

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS (5-run average)")
print("=" * 70)
print(f"{'Mode':<25s} {'ACC':>8s} {'PRE':>8s} {'SEN':>8s} {'SPE':>8s}")
print("-" * 70)
for mode, label in [
    ("eeg_only", "EEG-only (EEGNet)"),
    ("eeg_real", "EEG + Real fNIRS"),
    ("eeg_synth", "EEG + Synth fNIRS (SCDM)"),
]:
    vals = results[mode]
    avg = {k: np.mean([v[k] for v in vals]) for k in vals[0]}
    std = {k: np.std([v[k] for v in vals]) for k in vals[0]}
    print(f"{label:<25s} {avg['ACC']:>6.1f}%  {avg['PRE']:>6.1f}%  {avg['SEN']:>6.1f}%  {avg['SPE']:>6.1f}%")
    print(f"{'':25s} +/-{std['ACC']:>4.1f}%  +/-{std['PRE']:>4.1f}%  +/-{std['SEN']:>4.1f}%  +/-{std['SPE']:>4.1f}%")
sys.stdout.flush()
