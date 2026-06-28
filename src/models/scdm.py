"""Diffusion process + SCDM wrapper.

Clean conditioning: EEG (e0) is passed unnoised to the U-Net at all timesteps,
giving the model a strong cross-modal signal. Only fNIRS is diffused.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionProcess:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = "cpu", schedule: str = "linear"):
        self.T = T
        if schedule == "cosine":
            self.betas = self._cosine_schedule(T).to(device)
        else:
            self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _cosine_schedule(T, s=0.008):
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Closed-form forward diffusion with externally supplied (shared) noise."""
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


from .variants import ConfigurableUNet, SimpleDenoiser


class SCDM(nn.Module):
    """Ties the denoiser to the diffusion process for training, sampling and inference."""

    def __init__(self, diffusion: DiffusionProcess, base_channels: int = 32,
                 spatial: str = "scg_cross", temporal: str = "mtr",
                 eeg_hrf: bool = False):
        super().__init__()
        self.diffusion = diffusion
        if spatial == "simple":
            self.model = SimpleDenoiser(base_channels=base_channels,
                                        eeg_hrf=eeg_hrf)
        else:
            self.model = ConfigurableUNet(base_channels=base_channels,
                                          spatial=spatial, temporal=temporal,
                                          eeg_hrf=eeg_hrf)

    # -------------------------------------------------------------- training
    def loss(self, e0, f0, planes):
        B = e0.size(0)
        t = torch.randint(0, self.diffusion.T, (B,), device=e0.device)
        noise_f = torch.randn_like(f0)
        ft = self.diffusion.q_sample(f0, t, noise_f)
        noise_pred = self.model(e0, ft, planes, t)
        return F.mse_loss(noise_pred, noise_f)

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, e0, planes, stochastic: bool = True):
        """Generate synthetic fNIRS from EEG. e0: (B,30,4000)."""
        d = self.diffusion
        B = e0.size(0)
        ft = torch.randn(B, 36, 256, device=e0.device)
        for t in range(d.T - 1, -1, -1):
            tt = torch.full((B,), t, device=e0.device, dtype=torch.long)
            noise_pred = self.model(e0, ft, planes, tt)
            a_t, ab_t = d.alphas[t], d.alpha_bar[t]
            mean = (ft - (1 - a_t) / torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(a_t)
            if stochastic and t > 0:
                ft = mean + torch.sqrt(d.betas[t]) * torch.randn_like(ft)
            else:
                ft = mean
        return ft

    @torch.no_grad()
    def sample_ddim(self, e0, planes, steps: int = 100, eta: float = 0.0):
        """DDIM sampling — fewer steps, faster inference, same or better quality.
        steps: number of reverse steps (e.g. 50, 100, 200 instead of 1000).
        eta: 0 = deterministic (DDIM), 1 = stochastic (equivalent to DDPM).
        """
        d = self.diffusion
        B = e0.size(0)
        seq = torch.linspace(0, d.T - 1, steps + 1, dtype=torch.long, device=e0.device)
        seq = seq.flip(0)
        ft = torch.randn(B, 36, 256, device=e0.device)
        for i in range(len(seq) - 1):
            t_cur = seq[i]
            t_next = seq[i + 1]
            tt = torch.full((B,), t_cur, device=e0.device, dtype=torch.long)
            eps = self.model(e0, ft, planes, tt)
            ab_cur = d.alpha_bar[t_cur].clamp(min=1e-8)
            ab_next = d.alpha_bar[t_next].clamp(min=1e-8)
            x0_pred = (ft - torch.sqrt(1 - ab_cur) * eps) / torch.sqrt(ab_cur)
            x0_pred = x0_pred.clamp(-5, 5)
            sigma = eta * torch.sqrt(torch.clamp(
                (1 - ab_next) / (1 - ab_cur) * (1 - ab_cur / ab_next), min=0))
            dir_xt = torch.sqrt(torch.clamp(1 - ab_next - sigma ** 2, min=0)) * eps
            ft = torch.sqrt(ab_next) * x0_pred + dir_xt
            ft = torch.nan_to_num(ft, nan=0.0)
            if sigma > 0:
                ft = ft + sigma * torch.randn_like(ft)
        return ft
