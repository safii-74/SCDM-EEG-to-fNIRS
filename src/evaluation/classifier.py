"""EEGNet-based classifiers for evaluation.

Implements substitutes for ESNet/FGANet (Kwak 2022, not public):
  - EEGNet: EEG-only classifier (depthwise + separable temporal/spatial conv)
  - HybridNet: EEG + fNIRS two-branch classifier

Evaluation protocol: compare EEG-only, EEG+real fNIRS, EEG+synthetic fNIRS
under label ratios 2:8 to 8:2, average over 5 runs.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit

from .metrics import classification_metrics


class EEGNet(nn.Module):
    """EEGNet for motor imagery classification. Input: (B, 30, 4000)."""

    def __init__(self, n_channels: int = 30, n_samples: int = 4000,
                 n_classes: int = 2, F1: int = 8, D: int = 2, F2: int = 16,
                 downsample: int = 20):
        super().__init__()
        self.downsample = downsample
        n_in = n_samples // downsample  # 200

        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        feat_len = n_in // 32
        self._feat_dim = F2 * feat_len
        if n_classes is not None:
            self.fc = nn.Linear(self._feat_dim, n_classes)
        else:
            self.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample temporally for efficiency
        if self.downsample > 1:
            x = x[:, :, ::self.downsample]
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = self.separable(x)
        x = x.flatten(1)
        return self.fc(x)


class fNIRSBranch(nn.Module):
    """Small 1D CNN for fNIRS features. Input: (B, 36, 256)."""

    def __init__(self, n_channels: int = 36, n_samples: int = 256, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(128, feat_dim, 3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B, feat_dim)


class HybridNet(nn.Module):
    """Two-branch classifier: EEGNet branch + fNIRS CNN branch -> dense -> 2-way."""

    def __init__(self, n_eeg_ch: int = 30, n_eeg_samples: int = 4000,
                 n_fnirs_ch: int = 36, n_fnirs_samples: int = 256,
                 n_classes: int = 2, downsample: int = 20):
        super().__init__()
        self.eeg_branch = EEGNet(n_eeg_ch, n_eeg_samples, n_classes=None,
                                 downsample=downsample)
        eeg_feat_dim = self.eeg_branch._feat_dim

        self.fnirs_branch = fNIRSBranch(n_fnirs_ch, n_fnirs_samples, feat_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(eeg_feat_dim + 64, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> torch.Tensor:
        eeg_feat = self.eeg_branch(eeg)
        fnirs_feat = self.fnirs_branch(fnirs)
        combined = torch.cat([eeg_feat, fnirs_feat], dim=1)
        return self.classifier(combined)


def train_classifier(model, train_loader, epochs=50, lr=1e-3, device="cpu"):
    """Train a classifier and return the trained model."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for batch in train_loader:
            if len(batch) == 3:  # EEG + fNIRS + labels
                eeg, fnirs, labels = [b.to(device) for b in batch]
                logits = model(eeg, fnirs)
            else:  # EEG-only + labels
                eeg, labels = batch[0].to(device), batch[1].to(device)
                logits = model(eeg)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()
    return model


def evaluate_classifier(model, test_loader, device="cpu"):
    """Evaluate classifier, return metrics dict."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                eeg, fnirs, labels = [b.to(device) for b in batch]
                logits = model(eeg, fnirs)
            else:
                eeg, labels = batch[0].to(device), batch[1].to(device)
                logits = model(eeg)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return classification_metrics(y_true, y_pred)


def run_evaluation(eeg: np.ndarray, labels: np.ndarray,
                   fnirs_real: np.ndarray | None = None,
                   fnirs_synth: np.ndarray | None = None,
                   train_ratios: list[float] | None = None,
                   n_runs: int = 5, epochs: int = 50, batch_size: int = 32,
                   device: str = "cpu") -> dict:
    """Full evaluation protocol.

    Trains classifiers under different label ratios and compares:
      - EEG-only
      - EEG + real fNIRS
      - EEG + synthetic fNIRS

    Returns dict of results per mode per ratio.
    """
    if train_ratios is None:
        train_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = {}

    for ratio in train_ratios:
        ratio_key = f"{int(ratio*10)}:{int((1-ratio)*10)}"
        results[ratio_key] = {"eeg_only": [], "eeg_real": [], "eeg_synth": []}

        for run in range(n_runs):
            sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio,
                                        random_state=run * 42)
            train_idx, test_idx = next(sss.split(eeg, labels))

            # EEG-only
            eeg_train = torch.from_numpy(eeg[train_idx])
            eeg_test = torch.from_numpy(eeg[test_idx])
            lab_train = torch.from_numpy(labels[train_idx])
            lab_test = torch.from_numpy(labels[test_idx])

            train_ds = TensorDataset(eeg_train, lab_train)
            test_ds = TensorDataset(eeg_test, lab_test)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=batch_size)

            model_eeg = EEGNet()
            model_eeg = train_classifier(model_eeg, train_dl, epochs, device=device)
            metrics_eeg = evaluate_classifier(model_eeg, test_dl, device)
            results[ratio_key]["eeg_only"].append(metrics_eeg)

            # EEG + real fNIRS
            if fnirs_real is not None:
                fnirs_r_train = torch.from_numpy(fnirs_real[train_idx])
                fnirs_r_test = torch.from_numpy(fnirs_real[test_idx])
                train_ds_h = TensorDataset(eeg_train, fnirs_r_train, lab_train)
                test_ds_h = TensorDataset(eeg_test, fnirs_r_test, lab_test)
                train_dl_h = DataLoader(train_ds_h, batch_size=batch_size, shuffle=True)
                test_dl_h = DataLoader(test_ds_h, batch_size=batch_size)

                model_hybrid = HybridNet()
                model_hybrid = train_classifier(model_hybrid, train_dl_h, epochs, device=device)
                metrics_real = evaluate_classifier(model_hybrid, test_dl_h, device)
                results[ratio_key]["eeg_real"].append(metrics_real)

            # EEG + synthetic fNIRS
            if fnirs_synth is not None:
                fnirs_s_train = torch.from_numpy(fnirs_synth[train_idx])
                fnirs_s_test = torch.from_numpy(fnirs_synth[test_idx])
                train_ds_s = TensorDataset(eeg_train, fnirs_s_train, lab_train)
                test_ds_s = TensorDataset(eeg_test, fnirs_s_test, lab_test)
                train_dl_s = DataLoader(train_ds_s, batch_size=batch_size, shuffle=True)
                test_dl_s = DataLoader(test_ds_s, batch_size=batch_size)

                model_synth = HybridNet()
                model_synth = train_classifier(model_synth, train_dl_s, epochs, device=device)
                metrics_synth = evaluate_classifier(model_synth, test_dl_s, device)
                results[ratio_key]["eeg_synth"].append(metrics_synth)

    # Average across runs
    avg_results = {}
    for ratio_key, modes in results.items():
        avg_results[ratio_key] = {}
        for mode, runs in modes.items():
            if runs:
                avg_results[ratio_key][mode] = {
                    metric: np.mean([r[metric] for r in runs])
                    for metric in runs[0].keys()
                }
    return avg_results
