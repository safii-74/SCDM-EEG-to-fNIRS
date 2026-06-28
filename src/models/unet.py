"""SCDM U-Net backbone.

Both modalities enter as (B, 32, 256).  Down-blocks HALVE channels and length;
up-blocks DOUBLE both.  Channel ladder: 32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32.
Additive skip connections link matching encoder/decoder resolutions.

Interpretation note (paper is underspecified here): the running representation is
initialised by fusing the adapted noisy-fNIRS (f_t) and EEG (e_t); SCG then performs
the EEG->fNIRS correlation-guided channel mapping at each block.  The fusion conv is
intentionally swappable.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SCGModule, MTRModule


def sinusoidal_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)      # (B, dim)


class InputAdapter(nn.Module):
    """EEG and fNIRS -> common (out_ch, 256).

    eeg_hrf=False: EEG (30,4000) — strided convs + interpolate.
    eeg_hrf=True:  EEG (30,256)  — non-strided convs (already at fNIRS resolution).
    """

    def __init__(self, out_ch: int = 32, target_len: int = 256, eeg_hrf: bool = False):
        super().__init__()
        self.target_len = target_len
        self.eeg_hrf = eeg_hrf
        if eeg_hrf:
            self.eeg_t = nn.Sequential(
                nn.Conv1d(30, out_ch, 3, padding=1), nn.SiLU(),
                nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.SiLU(),
            )
        else:
            self.eeg_t = nn.Sequential(
                nn.Conv1d(30, out_ch, 7, stride=4, padding=3), nn.SiLU(),
                nn.Conv1d(out_ch, out_ch, 7, stride=4, padding=3), nn.SiLU(),
            )
        self.fnirs_c = nn.Conv1d(36, out_ch, 1)

    def forward(self, e: torch.Tensor, f: torch.Tensor):
        e = self.eeg_t(e)
        if not self.eeg_hrf:
            e = F.interpolate(e, size=self.target_len,
                              mode="linear", align_corners=False)
        return e, self.fnirs_c(f)


class SampleBlock(nn.Module):
    def __init__(self, channels: int, mode: str, time_dim: int = 128):
        super().__init__()
        d_out = channels // 2 if mode == "down" else channels * 2
        self.scg = SCGModule(channels, d_out, input_type="cross")
        self.mtr = MTRModule(d_out, mode)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, planes, t_emb):
        x = x + self.time_proj(t_emb)[..., None]            # broadcast over length
        x = self.scg(x, planes)                             # channel change
        return self.mtr(x)                                  # length change


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32, time_dim: int = 128):
        super().__init__()
        self.adapter = InputAdapter(out_ch=base_channels)
        self.fuse = nn.Conv1d(base_channels * 2, base_channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.time_dim = time_dim

        c = base_channels                                   # 32
        self.down = nn.ModuleList([
            SampleBlock(c, "down", time_dim),               # 32 -> 16
            SampleBlock(c // 2, "down", time_dim),          # 16 -> 8
            SampleBlock(c // 4, "down", time_dim),          # 8  -> 4
        ])
        self.up = nn.ModuleList([
            SampleBlock(c // 8, "up", time_dim),            # 4  -> 8
            SampleBlock(c // 4, "up", time_dim),            # 8  -> 16
            SampleBlock(c // 2, "up", time_dim),            # 16 -> 32
        ])
        self.out_conv = nn.Conv1d(c, 36, 1)                 # predicted fNIRS noise

    def forward(self, et, ft, planes, t):
        e, f = self.adapter(et, ft)                         # (B,32,256) each
        x = self.fuse(torch.cat([e, f], dim=1))             # (B,32,256)
        t_emb = self.time_mlp(sinusoidal_embed(t, self.time_dim))

        s0 = self.down[0](x, planes, t_emb)                 # (B,16,128)
        s1 = self.down[1](s0, planes, t_emb)                # (B,8,64)
        b = self.down[2](s1, planes, t_emb)                 # (B,4,32) bottleneck

        u = self.up[0](b, planes, t_emb) + s1               # (B,8,64)
        u = self.up[1](u, planes, t_emb) + s0               # (B,16,128)
        u = self.up[2](u, planes, t_emb)                    # (B,32,256)
        return self.out_conv(u)                             # (B,36,256)
