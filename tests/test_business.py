"""Tests for cutoff analysis and reject inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.business.cutoff import (
    optimal_cutoff,
    raroc_argmax_cutoff,
    risk_appetite_cutoff,
    sweep_cutoffs,
)
from credit_risk.business.reject_inference import parcelling


class TestCutoffSweep:
    def _make_data(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = 500
        y_true = rng.binomial(1, 0.15, n)
        # Scores anti-correlated with defaults: bads have lower scores
        y_score = rng.normal(600 - 100 * y_true, 50)
        ead = rng.uniform(5000, 30000, n)
        return y_true, y_score, ead

    def test_approval_rate_decreasing_with_threshold(self) -> None:
        y_true, y_score, ead = self._make_data()
        df = sweep_cutoffs(y_true, y_score, ead, n_thresholds=50)
        # Higher threshold → fewer approvals
        assert (df["approval_rate"].diff().dropna() <= 0.01).mean() >= 0.9

    def test_bad_rate_decreasing_with_threshold(self) -> None:
        y_true, y_score, ead = self._make_data()
        df = sweep_cutoffs(y_true, y_score, ead, n_thresholds=50)
        # Stricter cutoff → lower bad rate among approved
        non_zero = df[df["n_approved"] > 5].reset_index(drop=True)
        if len(non_zero) > 10:
            corr = non_zero["threshold"].corr(non_zero["bad_rate"])
            assert corr <= 0.0, "Bad rate should decrease as threshold increases"

    def test_profit_curve_has_interior_maximum(self) -> None:
        y_true, y_score, ead = self._make_data()
        df = sweep_cutoffs(y_true, y_score, ead, n_thresholds=100)
        opt = optimal_cutoff(df)
        # Optimal not at the extreme (approve all or approve none)
        assert opt["approval_rate"] > 0.05
        assert opt["approval_rate"] < 0.99

    def test_no_negative_rates(self) -> None:
        y_true, y_score, ead = self._make_data()
        df = sweep_cutoffs(y_true, y_score, ead)
        assert (df["approval_rate"] >= 0.0).all()
        assert (df["bad_rate"] >= 0.0).all()

    def test_approved_bad_plus_declined_bad_equals_n_bad(self) -> None:
        y_true, y_score, ead = self._make_data()
        df = sweep_cutoffs(y_true, y_score, ead, n_thresholds=20)
        n_bad = int(y_true.sum())
        for _, row in df.iterrows():
            assert int(row["approved_bad"]) + int(row["declined_bad"]) == n_bad

    def test_optimal_cutoff_keys(self) -> None:
        y_true, y_score, ead = self._make_data()
        df = sweep_cutoffs(y_true, y_score, ead)
        opt = optimal_cutoff(df)
        for key in ["threshold", "approval_rate", "bad_rate", "expected_profit"]:
            assert key in opt


class TestRejectInference:
    def _make_accepted_rejected(
        self, seed: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        n_acc = 200
        n_rej = 100

        acc = pd.DataFrame({
            "feature_a": rng.normal(0, 1, n_acc),
            "feature_b": rng.normal(0, 1, n_acc),
            "target": rng.binomial(1, 0.15, n_acc),
            "pd_pred": rng.uniform(0.05, 0.40, n_acc),
            "score": rng.normal(600, 50, n_acc),
        })

        rej = pd.DataFrame({
            "feature_a": rng.normal(-1, 1, n_rej),  # riskier profile
            "feature_b": rng.normal(-0.5, 1, n_rej),
            "pd_pred": rng.uniform(0.30, 0.70, n_rej),
            "score": rng.normal(530, 50, n_rej),
        })
        return acc, rej

    def test_parcelling_output_rows(self) -> None:
        acc, rej = self._make_accepted_rejected()
        combined = parcelling(acc, rej)
        # accepted rows + 2× rejected rows
        assert len(combined) == len(acc) + 2 * len(rej)

    def test_parcelling_weights_in_0_1(self) -> None:
        acc, rej = self._make_accepted_rejected()
        combined = parcelling(acc, rej)
        assert (combined["weight"] >= 0.0).all()
        assert (combined["weight"] <= 1.0).all()

    def test_parcelling_accepted_weight_is_1(self) -> None:
        acc, rej = self._make_accepted_rejected()
        combined = parcelling(acc, rej)
        acc_rows = combined[combined["_source"] == "accepted"]
        assert (acc_rows["weight"] == 1.0).all()

    def test_parcelling_bad_good_weights_sum_to_1(self) -> None:
        """For each rejected loan: weight_bad + weight_good = 1."""
        acc, rej = self._make_accepted_rejected()
        combined = parcelling(acc, rej)
        rej_good = combined[combined["_source"] == "rejected_good"]["weight"].values
        rej_bad = combined[combined["_source"] == "rejected_bad"]["weight"].values
        np.testing.assert_allclose(rej_good + rej_bad, 1.0, atol=1e-10)

    def test_parcelling_preserves_marginal_bad_rate(self) -> None:
        """Weighted bad rate in parcelled rejects ≈ mean PD of rejects."""
        acc, rej = self._make_accepted_rejected()
        combined = parcelling(acc, rej)
        rej_rows = combined[combined["_source"].str.startswith("rejected")]
        weighted_bad_rate = float(
            (rej_rows["target"] * rej_rows["weight"]).sum() / rej_rows["weight"].sum()
        )
        expected = float(rej["pd_pred"].mean())
        assert abs(weighted_bad_rate - expected) < 0.02

    def test_align_reject_data_robustness(self) -> None:
        from credit_risk.business.reject_inference import align_reject_data

        # 1. Prepare messy rejected data
        df_rejected = pd.DataFrame({
            "Loan-Amount": ["10000", "5000", None],
            "Risk-Score": ["710", "650", None],
            "Debt-To-Income-Ratio": ["15.5%", "22.3%", None],
            "Employment-Length": ["10+ years", "3 years", None],
            "Annual-Income": ["60000", "40000", None],
        })

        # 2. Prepare training data with standard schema and some extra variables to impute
        df_train = pd.DataFrame({
            "loan_amnt": [10000.0, 5000.0, 15000.0],
            "fico_range_low": [700.0, 640.0, 680.0],
            "fico_range_high": [704.0, 644.0, 684.0],
            "dti": [12.0, 20.0, 18.0],
            "emp_length": ["5 years", "3 years", "10+ years"],
            "annual_inc": [50000.0, 45000.0, 55000.0],
            # scorecard woe_variables to be imputed
            "home_ownership": ["MORTGAGE", "RENT", "OWN"],
            "purpose": ["credit_card", "debt_consolidation", "credit_card"],
            "revol_util": [45.5, 60.0, 30.0],
        })

        woe_variables = ["loan_amnt", "fico_range_low", "fico_range_high", "dti", "emp_length", "annual_inc", "home_ownership", "purpose", "revol_util"]

        # 3. Align reject data
        df_aligned = align_reject_data(df_rejected, df_train, woe_variables)

        # 4. Assert correctness
        # Row 0:
        assert df_aligned.loc[0, "loan_amnt"] == 10000.0
        assert df_aligned.loc[0, "fico_range_low"] == 710.0
        assert df_aligned.loc[0, "fico_range_high"] == 714.0
        assert df_aligned.loc[0, "dti"] == 15.5
        assert df_aligned.loc[0, "emp_length"] == "10+ years"
        assert df_aligned.loc[0, "annual_inc"] == 60000.0

        # Row 2 (missing values): should get defaults/fallbacks
        assert df_aligned.loc[2, "loan_amnt"] == 10000.0
        assert df_aligned.loc[2, "fico_range_low"] == 600.0
        assert df_aligned.loc[2, "fico_range_high"] == 604.0
        assert df_aligned.loc[2, "dti"] == 25.0
        assert df_aligned.loc[2, "emp_length"] == "< 1 year"
        assert df_aligned.loc[2, "annual_inc"] == 45000.0

        # Imputed scorecard variables (revol_util should be mean = 45.1667, purpose should be mode = credit_card)
        assert df_aligned["revol_util"].iloc[0] == pytest.approx(45.1667, abs=1e-3)
        assert df_aligned["purpose"].iloc[0] == "credit_card"

    def test_align_reject_data_no_risk_score(self) -> None:
        """align_reject_data must not raise KeyError if 'risk_score' absent."""
        from credit_risk.business.reject_inference import align_reject_data
        import pandas as pd
        df_rej = pd.DataFrame({"Amount Requested": [5000.0], "Debt-To-Income Ratio": ["15%"]})
        df_train = pd.DataFrame({"funded_amnt": [5000.0], "dti": [15.0], "int_rate": [10.0]})
        result = align_reject_data(df_rejected=df_rej, df_train=df_train, woe_variables=["funded_amnt"])
        assert "funded_amnt" in result.columns


class TestRiskAppetiteCutoff:
    @staticmethod
    def _rows(spec: list[tuple[int, float, float, float]]) -> list[dict]:
        """spec = (cutoff, approval_rate, bad_rate, raroc). Lower cutoff => more
        approvals and a higher approved bad rate (monotone, as on real data)."""
        return [
            {
                "cutoff": c, "approval_rate": a, "bad_rate": b, "raroc": rr,
                "expected_profit": a * 1e6, "capital_charge": a * 1e6, "expected_loss": 0.0,
            }
            for (c, a, b, rr) in spec
        ]

    def _grid(self) -> list[dict]:
        # bad rate rises as cutoff falls; profit/approval rise as cutoff falls (corner book)
        return self._rows([
            (400, 1.00, 0.2152, 0.5109),
            (500, 0.949, 0.2001, 0.4781),
            (520, 0.7696, 0.1638, 0.4157),
            (530, 0.6142, 0.1379, 0.3741),
            (540, 0.4375, 0.1100, 0.3136),
            (560, 0.1579, 0.0628, 0.1140),
        ])

    def test_interior_cutoff_within_appetite(self) -> None:
        # Ceiling 15%: cutoff 520 (16.38%) breaches, 530 (13.79%) is the most
        # inclusive within appetite -> operating cutoff = 530 (interior).
        opt = risk_appetite_cutoff(self._grid(), max_bad_rate=0.15)
        assert opt is not None
        assert opt["cutoff"] == 530
        assert opt["bad_rate"] <= 0.15
        # not the profit corner (400 / full approval)
        assert opt["approval_rate"] < 1.0

    def test_tighter_appetite_raises_cutoff(self) -> None:
        opt = risk_appetite_cutoff(self._grid(), max_bad_rate=0.12)
        assert opt is not None
        assert opt["cutoff"] == 540  # only <=11% and below qualify

    def test_no_cutoff_meets_ceiling_returns_none(self) -> None:
        assert risk_appetite_cutoff(self._grid(), max_bad_rate=0.01) is None

    def test_raroc_argmax_is_corner_on_priced_book(self) -> None:
        # Highest RAROC is at full approval (risk-priced book) -> disclosed corner.
        corner = raroc_argmax_cutoff(self._grid())
        assert corner is not None
        assert corner["cutoff"] == 400
        assert corner["approval_rate"] == 1.0
