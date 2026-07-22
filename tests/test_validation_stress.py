"""Stress tests for validation metrics: RAG status, Hosmer-Lemeshow, and vintage PD calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.validation.discrimination import RAGStatus
from credit_risk.validation.calibration import hosmer_lemeshow_test
from credit_risk.validation.backtest import vintage_pd_accuracy


class TestRAGStatusStress:
    def test_gini_degradation_zero_or_negative(self) -> None:
        """Test Gini degradation when there is zero or negative degradation (OOT improves)."""
        # Zero degradation
        rag = RAGStatus(gini_train=0.40, gini_oot=0.40, psi=0.05)
        assert rag.gini_rag == "GREEN"
        assert rag.overall == "GREEN"

        # Negative degradation (OOT improves)
        rag_imp = RAGStatus(gini_train=0.40, gini_oot=0.45, psi=0.05)
        assert rag_imp.gini_rag == "GREEN"
        assert rag_imp.overall == "GREEN"

    def test_psi_negative_or_extreme(self) -> None:
        """Test RAGStatus with negative or extreme PSI values."""
        # Negative PSI (e.g., due to numerical precision or bad input)
        rag_neg = RAGStatus(gini_train=0.40, gini_oot=0.38, psi=-0.01)
        assert rag_neg.psi_rag == "GREEN"
        assert rag_neg.overall == "GREEN"

        # Extreme PSI
        rag_ext = RAGStatus(gini_train=0.40, gini_oot=0.38, psi=99.0)
        assert rag_ext.psi_rag == "RED"
        assert rag_ext.overall == "RED"

    def test_exact_boundaries(self) -> None:
        """Test exact boundary values for Gini degradation and PSI."""
        # Gini degradation thresholds: AMBER = 0.05, RED = 0.10
        # d = 0.05 (exactly)
        rag_gini_boundary_amber = RAGStatus(gini_train=0.40, gini_oot=0.35, psi=0.05)
        assert rag_gini_boundary_amber.gini_rag == "AMBER"

        # d = 0.10 (exactly)
        rag_gini_boundary_red = RAGStatus(gini_train=0.40, gini_oot=0.30, psi=0.05)
        assert rag_gini_boundary_red.gini_rag == "RED"

        # PSI thresholds: GREEN < 0.10, AMBER < 0.25, RED >= 0.25
        # psi = 0.10 (exactly)
        rag_psi_boundary_amber = RAGStatus(gini_train=0.40, gini_oot=0.38, psi=0.10)
        assert rag_psi_boundary_amber.psi_rag == "AMBER"

        # psi = 0.25 (exactly)
        rag_psi_boundary_red = RAGStatus(gini_train=0.40, gini_oot=0.38, psi=0.25)
        assert rag_psi_boundary_red.psi_rag == "RED"


class TestHosmerLemeshowStress:
    def test_duplicate_predictions_df_mismatch(self) -> None:
        """Test that duplicate predictions reduce the number of unique bins, causing wrong df.
        
        This test checks if the returned degrees of freedom (df) matches the actual
        number of non-empty bins (G - 2) rather than the hardcoded (g - 2).
        """
        # Create predictions with only 3 unique values
        y_pred = np.array([0.1]*100 + [0.2]*100 + [0.3]*100)
        y_true = np.array([0]*90 + [1]*10 + [0]*80 + [1]*20 + [0]*70 + [1]*30)
        
        # Call HL test with g=10
        result = hosmer_lemeshow_test(y_true, y_pred, g=10)
        
        # With 3 unique values, decile_cuts has length at most 3.
        # This yields at most 3 groups. Therefore, df should be 3 - 2 = 1.
        # But if the code hardcodes df = g - 2 = 8, we verify the discrepancy.
        actual_groups = len(np.unique(np.digitize(y_pred, np.unique(np.percentile(y_pred, np.linspace(0, 100, 11)))[1:-1])))
        expected_df = actual_groups - 2
        
        # We assert that the returned df matches the true degrees of freedom
        assert result["df"] == expected_df, f"Returned df is {result['df']}, but expected {expected_df} based on actual groups ({actual_groups})"

    def test_zero_denominator_handling(self) -> None:
        """Test how HL handles zero expected default rates (zero denominator)."""
        # If y_pred has 0.0 (and is not clipped prior to hosmer_lemeshow_test),
        # e_g can be 0. If y_true contains defaults in that group, o_g > 0, leading to division by zero.
        y_pred = np.array([0.0]*50 + [0.5]*50)
        y_true = np.array([1]*10 + [0]*40 + [0]*25 + [1]*25)
        
        # Test if it runs without raising ZeroDivisionError (it should, due to 0 < e_g < n_g check)
        try:
            result = hosmer_lemeshow_test(y_true, y_pred, g=10)
        except ZeroDivisionError as exc:
            pytest.fail(f"hosmer_lemeshow_test raised ZeroDivisionError: {exc}")
            
        # However, because it skips the first group (since e_g == 0), the huge discrepancy (10 defaults when expecting 0)
        # is completely ignored, and h_stat only counts the second group.
        # Let's verify that h_stat is correctly high if the discrepancy was penalized.
        # It should be high (indicating poor fit), but because of the skip bug it will be low.
        # So we assert it is high to expose the bug.
        assert result["h_stat"] > 100.0, f"h_stat is {result['h_stat']}, which is too low because the mismatch in the e_g=0 group was ignored"

    def test_small_g_values(self) -> None:
        """Test HL behavior with g <= 2, which results in df <= 0."""
        y_pred = np.array([0.1]*50 + [0.5]*50)
        y_true = np.array([0]*50 + [1]*50)
        
        # If g=2, df = 2 - 2 = 0.
        result = hosmer_lemeshow_test(y_true, y_pred, g=2)
        assert result["df"] <= 0 or result["df"] == 0, f"Returned df: {result['df']}"


class TestVintagePDAccuracyStress:
    def test_zero_defaults_in_cohort(self) -> None:
        """Test vintage PD calibration flags when a cohort has 0 realised defaults."""
        df = pd.DataFrame({
            "issue_d": ["2015-01-01"] * 50,
            "pd_pred": [0.05] * 50,
            "target": [0] * 50,  # 0 defaults
        })
        result = vintage_pd_accuracy(df)
        
        # actual_dr is 0.0. Clipped actual_dr is 1e-6.
        # pd_ratio = 0.05 / 1e-6 = 50,000.
        # This will flag as "fail".
        assert result.loc[0, "calibration_flag"] == "fail"

    def test_perfect_zero_prediction_and_defaults(self) -> None:
        """Test when predicted PD and actual default rate are both exactly 0 (perfect fit)."""
        df = pd.DataFrame({
            "issue_d": ["2015-01-01"] * 50,
            "pd_pred": [0.0] * 50,  # model predicts 0 risk
            "target": [0] * 50,    # 0 defaults
        })
        result = vintage_pd_accuracy(df)
        
        # pd_ratio = 0.0 / 1e-6 = 0.0.
        # Flag is "fail" because 0.0 < 0.60.
        # But this is a perfect prediction, so it should ideally be "pass"!
        assert result.loc[0, "calibration_flag"] == "pass", f"Perfect zero risk prediction should pass, but got flag: {result.loc[0, 'calibration_flag']}"

    def test_vintage_pd_nan_handling(self) -> None:
        """Test vintage PD calibration with NaN values."""
        df = pd.DataFrame({
            "issue_d": ["2015-01-01"] * 50 + [np.nan] * 10,
            "pd_pred": [0.05] * 50 + [0.05] * 10,
            "target": [0] * 50 + [0] * 10,
        })
        # Should run without crashing by ignoring NaNs in vintage_col
        result = vintage_pd_accuracy(df)
        assert len(result) == 1
        assert "NaN" not in result["vintage"].values


class TestRAGStatusNaNStress:
    def test_rag_status_nan_inputs(self) -> None:
        """Test RAGStatus when some metrics are NaN."""
        # NaN in Gini OOT
        rag_nan_gini = RAGStatus(gini_train=0.40, gini_oot=np.nan, psi=0.05)
        # Should handle NaN gracefully (e.g. flagging RED or raising custom error, currently yields RED)
        assert rag_nan_gini.gini_rag == "RED"
        
        # NaN in PSI
        rag_nan_psi = RAGStatus(gini_train=0.40, gini_oot=0.40, psi=np.nan)
        assert rag_nan_psi.psi_rag == "RED"


class TestHosmerLemeshowNaNStress:
    def test_hosmer_lemeshow_nans_and_const(self) -> None:
        """Test HL test behavior with NaN predictions or constant predictions."""
        # Constant predictions (single unique value)
        y_pred = np.array([0.1] * 100)
        y_true = np.array([0] * 90 + [1] * 10)
        
        # Should not raise exception and should adjust degrees of freedom
        result = hosmer_lemeshow_test(y_true, y_pred, g=10)
        assert result["df"] <= 1

