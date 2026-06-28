"""Model variants: original U-Net with SCG/MTR, and simple ResConv denoiser."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SCGModule, MTRModule


class AttnControl(nn.Module):
    """Channel map d_in->d_out via 1x1 conv, then self-attention over time. Length fixed."""

    def __init__(self, d_in: int, d_out: int, heads: int = 1):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d_out, 1)
        h = heads if d_out % heads == 0 else 1
        self.attn = nn.MultiheadAttention(d_out, h, batch_first=True)

    def forward(self, value, planes=None):
        x = self.proj(value).transpose(1, 2)                # (B, L, d_out)
        x, _ = self.attn(x, x, x)
        return x.transpose(1, 2)                             # (B, d_out, L)


class CovControl(nn.Module):
    """Standard conv that halves (down) or doubles (up) length; channels unchanged."""

    def __init__(self, channels: int, mode: str = "down"):
        super().__init__()
        if mode == "down":
            self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


def build_spatial(kind: str, d_in: int, d_out: int):
    if kind == "attn":
        return AttnControl(d_in, d_out)
    # 'scg_cross' | 'scg_eeg' | 'scg_fnirs'
    return SCGModule(d_in, d_out, input_type=kind.split("_")[1])


def build_temporal(kind: str, channels: int, mode: str):
    return CovControl(channels, mode) if kind == "cov" else MTRModule(channels, mode)


class ConfigurableBlock(nn.Module):
    def __init__(self, channels, mode, spatial="scg_cross", temporal="mtr", time_dim=128):
        super().__init__()
        d_out = channels // 2 if mode == "down" else channels * 2
        self.spatial = build_spatial(spatial, channels, d_out)
        self.temporal = build_temporal(temporal, d_out, mode)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, planes, t_emb):
        x = x + self.time_proj(t_emb)[..., None]
        x = self.spatial(x, planes)
        return self.temporal(x)


class ConfigurableUNet(nn.Module):
    """U-Net identical in wiring to models.unet.UNet but with selectable modules.

    spatial in {'scg_cross','scg_eeg','scg_fnirs','attn'};  temporal in {'mtr','cov'}.
    Default ('scg_cross','mtr') reproduces the proposed SCDM.
    """

    def __init__(self, base_channels=32, time_dim=128,
                 spatial="scg_cross", temporal="mtr", eeg_hrf=False):
        super().__init__()
        from .unet import InputAdapter, sinusoidal_embed  # local import: no cycle
        self._sin = sinusoidal_embed
        self.time_dim = time_dim
        self.adapter = InputAdapter(out_ch=base_channels, eeg_hrf=eeg_hrf)
        self.fuse = nn.Conv1d(base_channels * 2, base_channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        c = base_channels
        self.down = nn.ModuleList([
            ConfigurableBlock(c, "down", spatial, temporal, time_dim),
            ConfigurableBlock(c // 2, "down", spatial, temporal, time_dim),
            ConfigurableBlock(c // 4, "down", spatial, temporal, time_dim)])
        self.up = nn.ModuleList([
            ConfigurableBlock(c // 8, "up", spatial, temporal, time_dim),
            ConfigurableBlock(c // 4, "up", spatial, temporal, time_dim),
            ConfigurableBlock(c // 2, "up", spatial, temporal, time_dim)])
        self.out_conv = nn.Conv1d(c, 36, 1)

    def forward(self, et, ft, planes, t):
        e, f = self.adapter(et, ft)
        x = self.fuse(torch.cat([e, f], dim=1))
        t_emb = self.time_mlp(self._sin(t, self.time_dim))
        s0 = self.down[0](x, planes, t_emb)
        s1 = self.down[1](s0, planes, t_emb)
        b = self.down[2](s1, planes, t_emb)
        u = self.up[0](b, planes, t_emb) + s1
        u = self.up[1](u, planes, t_emb) + s0
        u = self.up[2](u, planes, t_emb)
        return self.out_conv(u)


# ---------------------------------------------------------------------------
# Simple residual Conv1d denoiser — replaces the U-Net when spatial="simple"
# ---------------------------------------------------------------------------

def _sinusoidal_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[..., None]
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class PlanesEncoder(nn.Module):
    """Encode correlation planes into a conditioning vector.

    Processes the cross-modal (cef, cfe) and uni-modal (ce, cf) planes through
    small Conv2d networks, then projects to the conditioning dimension.
    """

    def __init__(self, cond_dim: int = 128):
        super().__init__()
        self.cross_enc = nn.Sequential(
            nn.Conv2d(30 + 36, 32, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.uni_enc = nn.Sequential(
            nn.Conv2d(30 + 36, 32, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, planes: dict) -> torch.Tensor:
        cross = torch.cat([planes["cef"], planes["cfe"]], dim=1)
        uni = torch.cat([planes["ce"], planes["cf"]], dim=1)
        h = torch.cat([self.cross_enc(cross), self.uni_enc(uni)], dim=1)
        return self.proj(h)


class SimpleDenoiser(nn.Module):
    """Residual ConvNet denoiser with cross-modal planes conditioning.

    Uses correlation planes as conditioning (added to time embedding) so every
    ResBlock sees both temporal AND spatial cross-modal information.

    eeg_hrf=True: EEG input is HRF-convolved features at fNIRS resolution (30, 256).
    eeg_hrf=False: Raw EEG (30, 4000). Stride-4 convs + interpolate (original).
    """

    def __init__(self, base_channels: int = 128, time_dim: int = 128,
                 n_blocks: int = 8, eeg_hrf: bool = False, **_ignored):
        super().__init__()
        c = base_channels
        self.time_dim = time_dim
        self.eeg_hrf = eeg_hrf
        if eeg_hrf:
            self.eeg_enc = nn.Sequential(
                nn.Conv1d(30, c, 3, padding=1), nn.SiLU(),
                nn.Conv1d(c, c, 3, padding=1), nn.SiLU(),
            )
        else:
            self.eeg_enc = nn.Sequential(
                nn.Conv1d(30, c, 7, stride=4, padding=3), nn.SiLU(),
                nn.Conv1d(c, c, 7, stride=4, padding=3), nn.SiLU(),
            )
        self.fnirs_proj = nn.Conv1d(36, c, 1)
        self.fuse = nn.Conv1d(c * 2, c, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.planes_enc = PlanesEncoder(cond_dim=time_dim)
        self.blocks = nn.ModuleList([ResBlock(c, time_dim) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, 36, 1))

    def forward(self, e0, ft, planes, t):
        e = self.eeg_enc(e0)
        if not self.eeg_hrf:
            e = F.interpolate(e, size=ft.size(-1),
                              mode="linear", align_corners=False)
        f = self.fnirs_proj(ft)
        x = self.fuse(torch.cat([e, f], dim=1))
        t_emb = self.time_mlp(_sinusoidal_embed(t, self.time_dim))
        p_emb = self.planes_enc(planes)
        cond = t_emb + p_emb
        for block in self.blocks:
            x = block(x, cond)
        return self.out(x)
