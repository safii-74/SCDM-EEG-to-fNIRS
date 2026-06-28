"""Shape and integration tests for SCDM. Run: PYTHONPATH=. python3 tests/test_shapes.py"""
import numpy as np
import torch

from src.models.modules import SCGModule, MTRModule
from src.models.unet import UNet
from src.models.scdm import SCDM, DiffusionProcess
from src.models.variants import ConfigurableUNet
from src.data.correlations import (build_coords16, make_correlation_planes,
                                    pearson_matrix, distance_correlation_matrix)

B = 2
PLANES = {k: torch.randn(B, c, 16, 16) for k, c in
          (("cef", 30), ("cfe", 36), ("ce", 30), ("cf", 36))}


def test_scg():
    assert SCGModule(32, 16, "cross")(torch.randn(B, 32, 256), PLANES).shape == (B, 16, 256)
    assert SCGModule(8, 16, "cross")(torch.randn(B, 8, 64), PLANES).shape == (B, 16, 64)


def test_mtr():
    for L in (256, 128, 64):
        assert MTRModule(8, "down")(torch.randn(B, 8, L)).shape == (B, 8, L // 2)
    for L in (32, 64, 128):
        assert MTRModule(8, "up")(torch.randn(B, 8, L)).shape == (B, 8, L * 2)


def test_unet():
    out = UNet()(torch.randn(B, 30, 4000), torch.randn(B, 36, 256), PLANES,
                 torch.randint(0, 1000, (B,)))
    assert out.shape == (B, 36, 256) and torch.isfinite(out).all()


def test_all_ablations():
    combos = [("attn", "cov"), ("attn", "mtr"), ("scg_eeg", "cov"), ("scg_eeg", "mtr"),
              ("scg_fnirs", "cov"), ("scg_fnirs", "mtr"), ("scg_cross", "mtr")]
    e, f, t = (torch.randn(B, 30, 4000), torch.randn(B, 36, 256),
               torch.randint(0, 1000, (B,)))
    for s, tm in combos:
        out = ConfigurableUNet(spatial=s, temporal=tm)(e, f, PLANES, t)
        assert out.shape == (B, 36, 256) and torch.isfinite(out).all(), (s, tm)


def test_train_and_sample():
    scdm = SCDM(DiffusionProcess(T=20))
    opt = torch.optim.Adam(scdm.parameters(), 1e-4)
    e0, f0 = torch.randn(B, 30, 4000), torch.randn(B, 36, 256)
    l0 = scdm.loss(e0, f0, PLANES); l0.backward(); opt.step()
    assert torch.isfinite(l0)
    syn = scdm.sample(e0, PLANES)
    assert syn.shape == (B, 36, 256) and torch.isfinite(syn).all()


def test_simple_denoiser_hrf():
    from src.models.variants import SimpleDenoiser
    sd = SimpleDenoiser(base_channels=64, eeg_hrf=True)
    e_hrf = torch.randn(B, 30, 256)
    f = torch.randn(B, 36, 256)
    t = torch.randint(0, 200, (B,))
    out = sd(e_hrf, f, PLANES, t)
    assert out.shape == (B, 36, 256) and torch.isfinite(out).all()


def test_scdm_hrf_train_and_sample():
    scdm = SCDM(DiffusionProcess(T=20), spatial="simple", eeg_hrf=True)
    opt = torch.optim.Adam(scdm.parameters(), 1e-4)
    e0 = torch.randn(B, 30, 256)
    f0 = torch.randn(B, 36, 256)
    l0 = scdm.loss(e0, f0, PLANES); l0.backward(); opt.step()
    assert torch.isfinite(l0)
    syn = scdm.sample_ddim(e0, PLANES, steps=5)
    assert syn.shape == (B, 36, 256) and torch.isfinite(syn).all()


def test_correlation_planes_realdcor():
    # Small signals so distance correlation is fast, real coords mapping.
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((30, 120)).astype(np.float32)
    fnirs = rng.standard_normal((36, 40)).astype(np.float32)
    ec = build_coords16(rng.standard_normal((30, 2)))
    fc = build_coords16(rng.standard_normal((36, 2)))
    planes = make_correlation_planes(eeg, fnirs, ec, fc)
    assert planes["cef"].shape == (30, 16, 16)
    assert planes["cfe"].shape == (36, 16, 16)
    assert planes["ce"].shape == (30, 16, 16)
    assert planes["cf"].shape == (36, 16, 16)
    # Pearson diagonal == 1, distance correlation in [0,1]
    assert np.allclose(np.diag(pearson_matrix(eeg)), 1.0, atol=1e-4)
    dc = distance_correlation_matrix(eeg, fnirs)
    assert dc.min() >= -1e-6 and dc.max() <= 1 + 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
