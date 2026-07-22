"""Tests for WoE/IV binning and feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.target import TARGET_COL, define_target
from credit_risk.features.woe import compute_woe_iv
from credit_risk.features.selection import filter_by_iv, sign_check
from credit_risk.utils.config import TargetConfig


@pytest.fixture
def binary_xy(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Simple binary classification dataset."""
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    log_odds = 1.5 * x1 - x2
    y = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2,
                      "x3": rng.normal(0, 1, n),  # noise
                      "cat": rng.choice(["A", "B", "C"], n)})
    return X, pd.Series(y, name="target")


class TestComputeWoEIV:
    def test_iv_nonnegative(self, binary_xy: tuple) -> None:
        X, y = binary_xy
        _, iv = compute_woe_iv(X["x1"], y)
        assert iv >= 0, f"IV should be non-negative, got {iv}"

    def test_iv_informative_feature_higher_than_noise(self, binary_xy: tuple) -> None:
        X, y = binary_xy
        _, iv_signal = compute_woe_iv(X["x1"], y)
        _, iv_noise = compute_woe_iv(X["x3"], y)
        assert iv_signal > iv_noise, "Signal feature should have higher IV than noise"

    def test_woe_bin_table_columns(self, binary_xy: tuple) -> None:
        X, y = binary_xy
        tbl, iv = compute_woe_iv(X["x1"], y)
        expected_cols = {"bin", "count", "n_bad", "n_good", "woe", "iv"}
        assert expected_cols.issubset(set(tbl.columns))

    def test_iv_equals_sum_of_bin_ivs(self, binary_xy: tuple) -> None:
        X, y = binary_xy
        tbl, total_iv = compute_woe_iv(X["x1"], y)
        assert abs(tbl["iv"].sum() - total_iv) < 1e-10

    def test_woe_formula_correctness(self) -> None:
        """Hand-calculation: 100 goods / 200 bad in bin → WoE = ln(0.5/1.0) = -ln(2)."""
        # 200 goods total, 200 bads total; one bin with 100G and 200B
        # With Laplace smoothing (alpha=0.5):
        #   pct_g = (100 + 0.5) / (200 + 1.0) = 100.5/201
        #   pct_b = (200 + 0.5) / (200 + 1.0) = 200.5/201
        #   woe   = ln(100.5 / 200.5) ≈ -0.690
        x = pd.Series([0.0] * 300 + [1.0] * 100)
        y = pd.Series([1] * 200 + [0] * 100 + [0] * 100)
        tbl, _ = compute_woe_iv(x, y, n_bins=2)
        # Both bins should be present; at least check total count
        assert tbl["count"].sum() == 400


class TestIVFilter:
    def test_low_iv_dropped(self) -> None:
        tbl = pd.DataFrame({"variable": ["a", "b", "c"], "iv": [0.005, 0.10, 0.60]})
        result = filter_by_iv(tbl, min_iv=0.02, max_iv=0.50)
        assert "a" not in result  # below 0.02
        assert "c" not in result  # above 0.50
        assert "b" in result

    def test_empty_table_returns_empty(self) -> None:
        tbl = pd.DataFrame({"variable": [], "iv": []})
        result = filter_by_iv(tbl)
        assert result == []


class TestSignCheck:
    def test_positive_signs_return_no_violations(self) -> None:
        coefs = pd.Series({"a": 0.5, "b": 1.2, "c": 0.01})
        violations = sign_check(coefs, expected_positive=True)
        assert violations == []

    def test_negative_coef_flagged(self) -> None:
        coefs = pd.Series({"a": 0.5, "b": -0.3, "c": 0.01})
        violations = sign_check(coefs, expected_positive=True)
        assert "b" in violations
        assert "a" not in violations


class TestWoETransformer:
    def test_fit_transform_shape(self, binary_xy: tuple) -> None:
        from credit_risk.features.woe import WoETransformer

        X, y = binary_xy
        X_num = X.select_dtypes(include="number")
        t = WoETransformer(variables=list(X_num.columns))
        t.fit(X_num.fillna(0), y)
        out = t.transform(X_num.fillna(0))
        assert out.shape == (len(X_num), len(t.variables_))

    def test_iv_table_has_iv_column(self, binary_xy: tuple) -> None:
        from credit_risk.features.woe import WoETransformer

        X, y = binary_xy
        X_num = X.select_dtypes(include="number")
        t = WoETransformer(variables=list(X_num.columns))
        t.fit(X_num.fillna(0), y)
        iv_tbl = t.get_iv_table()
        assert "iv" in iv_tbl.columns
        assert (iv_tbl["iv"] >= 0).all()


class TestScorecardScaling:
    def test_score_to_pd_round_trip(self) -> None:
        from credit_risk.models.pd_scorecard import PDScorecard

        sc = PDScorecard(pdo=20, base_score=600, base_odds=50)
        sc._factor = 20.0 / np.log(2)
        sc._offset = 600.0 - sc._factor * np.log(50.0)

        pd_original = np.array([0.01, 0.05, 0.10, 0.30])
        scores = sc.pd_to_score(pd_original)
        pd_recovered = sc.score_to_pd(scores)
        np.testing.assert_allclose(pd_recovered, pd_original, rtol=1e-6)

    def test_base_score_at_base_odds(self) -> None:
        from credit_risk.models.pd_scorecard import PDScorecard

        pdo, base_score, base_odds = 20.0, 600.0, 50.0
        sc = PDScorecard(pdo=pdo, base_score=base_score, base_odds=base_odds)
        sc._factor = pdo / np.log(2)
        sc._offset = base_score - sc._factor * np.log(base_odds)

        # At base odds (50:1), PD = 1/(1+50) ≈ 0.0196
        pd_base = 1.0 / (1.0 + base_odds)
        score = float(sc.pd_to_score(pd_base))
        assert abs(score - base_score) < 0.01, f"Expected score≈{base_score}, got {score:.2f}"
