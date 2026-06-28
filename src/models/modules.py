"""Core SCDM modules: SCG (spatial cross-modal) and MTR (multi-scale temporal).

Corrections applied vs. the original plan:
  * SCG: attention output uses score^T @ value so the output carries d_out
    (fNIRS-representation) channels, not d_in.
  * SCG: consumes PRECOMPUTED 16x16 correlation planes (no dcor inside forward).
  * MTR: causal convolution uses left-only zero padding via F.pad (the symmetric
    `padding` arg of Conv1d cannot do this); net length change is exactly 2x.
  * MTR: two parallel branches (depthwise->causal-dilated, and point-wise) fused
    at the end, matching Fig. 1, rather than a single sequential chain.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- SCG
class SCGModule(nn.Module):
    """Maps the channel dimension d_in -> d_out, keeping sequence length fixed.

    value:  (B, d_in, L)  running representation (serves as V).
    planes: dict of precomputed 16x16 correlation planes, each (B, C, 16, 16).
    """

    def __init__(self, d_in: int, d_out: int, input_type: str = "cross"):
        super().__init__()
        assert input_type in ("cross", "eeg", "fnirs")
        self.input_type = input_type
        q_ch = 30 if input_type in ("cross", "eeg") else 36
        k_ch = 36 if input_type in ("cross", "fnirs") else 30
        self.q_conv = nn.Conv2d(q_ch, d_in, kernel_size=3, padding=1)
        self.k_conv = nn.Conv2d(k_ch, d_out, kernel_size=3, padding=1)
        self.scale = d_out ** -0.5

    def _planes(self, planes):
        if self.input_type == "cross":
            return planes["cef"], planes["cfe"]
        if self.input_type == "eeg":
            return planes["ce"], planes["ce"]
        return planes["cf"], planes["cf"]

    def forward(self, value: torch.Tensor, planes: dict) -> torch.Tensor:
        pq, pk = self._planes(planes)                       # (B,30/36,16,16)
        Q = self.q_conv(pq).flatten(2)                      # (B, d_in, 256)
        K = self.k_conv(pk).flatten(2)                      # (B, d_out, 256)
        score = F.softmax((Q @ K.transpose(-1, -2)) * self.scale, dim=-1)  # (B,d_in,d_out)
        return score.transpose(-1, -2) @ value              # (B, d_out, L)


# ----------------------------------------------------------------------------- MTR
def causal_pad(x: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    """Left-only zero padding for causal 1D convolution."""
    return F.pad(x, ((kernel - 1) * dilation, 0))


class MTRModule(nn.Module):
    """Halves (down) or doubles (up) sequence length; channel count unchanged."""

    def __init__(self, channels: int, mode: str = "down"):
        super().__init__()
        assert mode in ("down", "up")
        self.mode = mode

        # Branch A: multi-scale depthwise (stride 1) -> fuse -> causal dilated.
        self.depthwise = nn.ModuleList(
            nn.Conv1d(channels, channels, k, stride=1, padding=k // 2, groups=channels)
            for k in (3, 5, 7, 9)
        )
        self.fuse_a = nn.Conv1d(channels * 4, channels, 1)
        # In DOWN mode the first causal conv has stride 2 (length /2); the others use
        # dilation only. In UP mode all causal convs are stride 1 (length preserved);
        # the x2 expansion is done afterwards by the transposed conv / interpolation.
        first_stride = 2 if mode == "down" else 1
        self.causal = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size=2,
                      stride=(first_stride if i == 0 else 1), dilation=d, groups=channels)
            for i, d in enumerate((1, 2, 4))
        )

        # Branch B: multi-scale point-wise (down) or transposed/interp (up)
        if mode == "down":
            self.pointwise = nn.ModuleList(
                nn.Conv1d(channels, channels, 1, stride=2) for _ in range(4)
            )
        else:
            self.tconv = nn.ConvTranspose1d(channels, channels, 2, stride=2)
            self.interp = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.out = nn.Conv1d(channels * 2, channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch A: depthwise multi-scale -> fuse -> causal dilated stack
        a = self.fuse_a(torch.cat([c(x) for c in self.depthwise], dim=1))
        for c in self.causal:
            a = self.act(c(causal_pad(a, 2, c.dilation[0])))
        # Branch B + length change to match target (down: /2, up: x2)
        if self.mode == "down":
            b = sum(pw(x) for pw in self.pointwise) / len(self.pointwise)
        else:
            a = (self.tconv(a) + self.interp(a)) / 2     # length x2
            b = self.interp(x)                            # length x2
        L = min(a.size(-1), b.size(-1))                   # defensive align
        return self.out(torch.cat([a[..., :L], b[..., :L]], dim=1))
