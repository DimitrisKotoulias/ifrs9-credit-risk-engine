"""Tests for EL, Basel IRB, and related risk calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


class TestBaselIRB:
    def test_pd_floor_enforced(self) -> None:
        from credit_risk.risk.basel_irb import irb_capital_requirement

        pd_below_floor = np.array([0.0001, 0.0002])
        pd_at_floor = np.array([0.0003, 0.0003])
        lgd = np.array([0.45, 0.45])

        k_below = irb_capital_requirement(pd_below_floor, lgd, pd_floor=0.0003)
        k_floor = irb_capital_requirement(pd_at_floor, lgd, pd_floor=0.0003)
        np.testing.assert_allclose(k_below, k_floor, rtol=1e-10)

    def test_correlation_bounded_0_03_to_0_16(self) -> None:
        from credit_risk.risk.basel_irb import irb_correlation

        pds = np.array([0.0001, 0.01, 0.10, 0.50, 0.99])
        r = irb_correlation(pds)
        assert (r >= 0.03 - 1e-10).all(), f"R below 0.03: {r}"
        assert (r <= 0.16 + 1e-10).all(), f"R above 0.16: {r}"

    def test_k_nonnegative(self) -> None:
        from credit_risk.risk.basel_irb import irb_capital_requirement

        pd_vals = np.array([0.01, 0.05, 0.10, 0.20])
        lgd_vals = np.full(4, 0.45)
        k = irb_capital_requirement(pd_vals, lgd_vals)
        assert (k >= 0.0).all()

    def test_rwa_equals_k_times_12_5_times_ead(self) -> None:
        from credit_risk.risk.basel_irb import irb_capital_requirement, irb_rwa

        pd_vals = np.array([0.01, 0.05])
        lgd_vals = np.array([0.45, 0.40])
        ead_vals = np.array([10000.0, 5000.0])
        k = irb_capital_requirement(pd_vals, lgd_vals)
        rwa = irb_rwa(pd_vals, lgd_vals, ead_vals)
        expected_rwa = k * 12.5 * ead_vals
        np.testing.assert_allclose(rwa, expected_rwa, rtol=1e-10)

    def test_bcbs_worked_example(self) -> None:
        """Verify against a hand-calculated BCBS-style example.

        For PD=0.01, LGD=0.45, retail other-retail:
        R = 0.03 * (1 - e^(-35*0.01)) / (1 - e^(-35)) + 0.16 * [1 - that]
        K = 0.45 * N[(1-R)^(-0.5) * G(0.01) + (R/(1-R))^0.5 * G(0.999)] - 0.01 * 0.45
        """
        from credit_risk.risk.basel_irb import irb_capital_requirement, irb_correlation

        pd_val = 0.01
        lgd_val = 0.45
        pd_floor = 0.0003

        # Hand-compute R
        e_neg35 = np.exp(-35.0)
        weight = (1 - np.exp(-35 * pd_val)) / (1 - e_neg35)
        r_expected = 0.03 * weight + 0.16 * (1 - weight)
        r_actual = float(irb_correlation(np.array([pd_val]))[0])
        assert abs(r_actual - r_expected) < 1e-10

        # Hand-compute K
        g_pd = float(norm.ppf(pd_val))
        g_999 = float(norm.ppf(0.999))
        inner = (1 - r_expected) ** (-0.5) * g_pd + (r_expected / (1 - r_expected)) ** 0.5 * g_999
        k_expected = float(lgd_val * norm.cdf(inner) - pd_val * lgd_val)

        k_actual = float(irb_capital_requirement(np.array([pd_val]), np.array([lgd_val]), pd_floor)[0])
        assert abs(k_actual - k_expected) < 1e-10, (
            f"K mismatch: actual={k_actual:.6f}, expected={k_expected:.6f}"
        )

    def test_higher_pd_gives_higher_k(self) -> None:
        from credit_risk.risk.basel_irb import irb_capital_requirement

        pd_low = np.array([0.01])
        pd_high = np.array([0.10])
        lgd = np.array([0.45])
        k_low = irb_capital_requirement(pd_low, lgd)
        k_high = irb_capital_requirement(pd_high, lgd)
        assert k_high > k_low, "Higher PD should give higher capital requirement"

    def test_run_basel_irb_adds_rwa_column(self) -> None:
        from credit_risk.risk.basel_irb import run_basel_irb

        df = pd.DataFrame({
            "pd_pred": [0.01, 0.05, 0.10],
            "ead": [10000.0, 5000.0, 8000.0],
        })
        result = run_basel_irb(df, lgd_downturn=0.45)
        assert "rwa" in result.columns
        assert "capital_requirement_k" in result.columns
        assert (result["rwa"] >= 0.0).all()

    def test_irb_correlation_formula(self):
        from credit_risk.risk.basel_irb import irb_correlation
        import numpy as np
        R = irb_correlation(np.array([0.01]))
        assert abs(R[0] - 0.12170) < 0.001, f"Got R={R[0]:.5f}"

    def test_irb_capital_non_negative(self):
        from credit_risk.risk.basel_irb import irb_capital_requirement
        import numpy as np
        K = irb_capital_requirement(np.array([0.001,0.01,0.1,0.5,0.99]), np.full(5, 0.45))
        assert np.all(K >= 0)

    def test_irb_correlation_bounds(self):
        from credit_risk.risk.basel_irb import irb_correlation
        import numpy as np
        R = irb_correlation(np.linspace(0.001, 0.999, 100))
        assert np.all(R >= 0.029) and np.all(R <= 0.161)


class TestExpectedLoss:
    def test_el_formula(self) -> None:
        from credit_risk.risk.expected_loss import compute_expected_loss

        df = pd.DataFrame({
            "pd_pred": [0.05, 0.10],
            "lgd_pred": [0.40, 0.45],
            "ead": [10000.0, 5000.0],
        })
        el = compute_expected_loss(df)
        assert abs(float(el.iloc[0]) - 0.05 * 0.40 * 10000.0) < 0.01
        assert abs(float(el.iloc[1]) - 0.10 * 0.45 * 5000.0) < 0.01

    def test_el_nonnegative(self, rng: np.random.Generator) -> None:
        from credit_risk.risk.expected_loss import compute_expected_loss

        n = 200
        df = pd.DataFrame({
            "pd_pred": rng.random(n) * 0.3,
            "lgd_pred": rng.random(n) * 0.8,
            "ead": rng.random(n) * 50000,
        })
        el = compute_expected_loss(df)
        assert (el >= 0.0).all()

    def test_el_rate_bounded(self, rng: np.random.Generator) -> None:
        from credit_risk.risk.expected_loss import compute_expected_loss

        n = 100
        df = pd.DataFrame({
            "pd_pred": np.clip(rng.random(n), 0, 1),
            "lgd_pred": np.clip(rng.random(n), 0, 1),
            "ead": rng.random(n) * 50000 + 1000,
        })
        el = compute_expected_loss(df)
        el_rate = el / df["ead"]
        assert (el_rate >= 0.0).all()
        assert (el_rate <= 1.0).all(), "EL rate cannot exceed 100%"
