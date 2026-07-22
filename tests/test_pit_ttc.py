"""Tests for the PiT vs TTC Vasicek decomposition."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from credit_risk.risk.pit_ttc import decompose_pit_ttc, run_pit_ttc


def test_constant_dr_gives_constant_z():
    """A flat default-rate series maps to a single constant systematic factor.

    Note: that constant is not zero — under the Vasicek model DR is convex in Z, so the
    TTC-average default rate corresponds to a mildly adverse Z (Jensen effect).
    """
    dr = np.full(20, 0.05)
    ttc, z = decompose_pit_ttc(dr, rho=0.15)
    assert abs(ttc - 0.05) < 1e-9
    assert np.allclose(z, z[0], atol=1e-9)  # all identical
    assert np.isfinite(z).all()


def test_ttc_is_mean_of_dr():
    dr = np.array([0.02, 0.04, 0.06, 0.08])
    ttc, _ = decompose_pit_ttc(dr)
    assert abs(ttc - dr.mean()) < 1e-12


def test_high_dr_gives_negative_z():
    """A quarter with a default rate above the TTC average is adverse (Z < 0)."""
    dr = np.array([0.02, 0.02, 0.02, 0.10])  # last quarter is a spike
    _, z = decompose_pit_ttc(dr)
    assert z[-1] < 0.0
    assert z[0] > 0.0  # below-average quarters are benign


def test_inversion_roundtrips():
    """Reconstructing DR from (TTC, Z) via the Vasicek formula recovers the input."""
    rng = np.random.default_rng(0)
    dr = np.clip(rng.uniform(0.01, 0.15, 30), 1e-6, 1 - 1e-6)
    rho = 0.15
    ttc, z = decompose_pit_ttc(dr, rho=rho)
    from scipy.special import ndtri  # noqa: PLC0415
    recon = ndtr((ndtri(ttc) - np.sqrt(rho) * z) / np.sqrt(1 - rho))
    assert np.allclose(recon, dr, atol=1e-9)


def test_empty_series():
    ttc, z = decompose_pit_ttc(np.array([]))
    assert ttc == 0.0 and z.size == 0


def test_rho_out_of_range_raises():
    with pytest.raises(ValueError):
        decompose_pit_ttc(np.array([0.05, 0.06]), rho=0.0)


def test_run_pit_ttc_dict_shape():
    df = pd.DataFrame({
        "quarter": [f"2010Q{i%4+1}" for i in range(12)],
        "default_rate": np.linspace(0.02, 0.08, 12),
    })
    out = run_pit_ttc(df, rho=0.15)
    assert set(out) == {"ttc_pd", "rho", "quarters", "default_rates", "z_factors"}
    assert len(out["z_factors"]) == 12
    assert len(out["quarters"]) == 12
