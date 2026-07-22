"""Challenge tests to stress-test validation metrics: RAGStatus, Hosmer-Lemeshow, and Vintage PD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as _scipy_stats

from credit_risk.validation.discrimination import RAGStatus
from credit_risk.validation.calibration import hosmer_lemeshow_test
from credit_risk.validation.backtest import vintage_pd_accuracy


def test_rag_status_boundaries() -> None:
    # 1. Gini degradation exactly at Amber threshold (0.05)
    # d = 0.40 - 0.35 = 0.05
    # return "GREEN" if d < 0.05 else ("AMBER" if d < 0.10 else "RED")
    # Expected: "AMBER" because 0.05 is not < 0.05.
    rag_amber = RAGStatus(gini_train=0.40, gini_oot=0.35, psi=0.0)
    assert rag_amber.gini_rag == "AMBER"

    # 2. Gini degradation exactly at Red threshold (0.10)
    # d = 0.40 - 0.30 = 0.10
    # Expected: "RED" because 0.10 is not < 0.10.
    rag_red = RAGStatus(gini_train=0.40, gini_oot=0.30, psi=0.0)
    assert rag_red.gini_rag == "RED"

    # 3. Negative Gini degradation (OOT has higher Gini than Train)
    # d = 0.35 - 0.40 = -0.05
    # Expected: "GREEN"
    rag_neg = RAGStatus(gini_train=0.35, gini_oot=0.40, psi=0.0)
    assert rag_neg.gini_rag == "GREEN"

    # 4. PSI exactly at Green/Amber threshold (0.10)
    # Expected: "AMBER" because 0.10 is not < 0.10.
    rag_psi_amber = RAGStatus(gini_train=0.40, gini_oot=0.40, psi=0.10)
    assert rag_psi_amber.psi_rag == "AMBER"

    # 5. PSI exactly at Amber/Red threshold (0.25)
    # Expected: "RED" because 0.25 is not < 0.25.
    rag_psi_red = RAGStatus(gini_train=0.40, gini_oot=0.40, psi=0.25)
    assert rag_psi_red.psi_rag == "RED"

    # 6. Negative PSI (extreme input)
    # Expected: "GREEN"
    rag_psi_neg = RAGStatus(gini_train=0.40, gini_oot=0.40, psi=-0.05)
    assert rag_psi_neg.psi_rag == "GREEN"


def test_hosmer_lemeshow_df_bug() -> None:
    # After the fix, HL test now uses DYNAMIC df based on actual non-empty groups.
    # y_pred has only 2 unique values → only 2 groups → df = 2 - 2 = 0.
    rng = np.random.default_rng(42)
    y_pred = np.array([0.1] * 50 + [0.9] * 50)
    y_true = rng.binomial(1, y_pred).astype(float)
    
    # We run HL test with default g=10.
    # The actual number of groups is 2 (since there are only 2 unique prediction values).
    # After fix: degrees of freedom = max(0, actual_non_empty_groups - 2) = max(0, 2-2) = 0.
    res = hosmer_lemeshow_test(y_true, y_pred, g=10)
    assert res["df"] == 0, f"After fix, expected df=0 for 2 groups, got df={res['df']}"
    
    # If g = 2, should also give df = 0 (not negative).
    res_g2 = hosmer_lemeshow_test(y_true, y_pred, g=2)
    assert res_g2["df"] == 0
    assert res_g2["p_value"] == 0.0 or res_g2["p_value"] == 1.0
    
    # If g = 1, df = max(0, 1-2) = 0 (clamped, not -1).
    res_g1 = hosmer_lemeshow_test(y_true, y_pred, g=1)
    assert res_g1["df"] == 0


def test_vintage_pd_calibration_flags_zero_defaults() -> None:
    # Cohort 1: perfect calibration at 0.0 default rate (both pred and actual are 0.0)
    # After fix: this correctly returns 'pass' (not 'fail')
    # Cohort 2: normal case
    df = pd.DataFrame({
        "issue_d": ["2015-01-01"] * 100 + ["2016-01-01"] * 100,
        "pd_pred": [0.0] * 100 + [0.10] * 100,
        "target": [0] * 100 + [1] * 10 + [0] * 90,
    })
    
    result = vintage_pd_accuracy(df, pd_col="pd_pred", target_col="target", vintage_col="issue_d")
    
    # Map to dict for validation
    flags = {row["vintage"]: row["calibration_flag"] for _, row in result.iterrows()}
    ratios = {row["vintage"]: row["pd_ratio"] for _, row in result.iterrows()}
    
    # For 2015Q1, predicted_pd is 0.0, actual_dr is 0.0.
    # After fix: perfect zero prediction correctly flags as 'pass'.
    assert ratios["2015Q1"] == 0.0
    assert flags["2015Q1"] == "pass", (
        f"Zero predicted / zero actual should be 'pass' (perfect fit), got '{flags['2015Q1']}'"
    )
    
    # For 2016Q1, predicted_pd is 0.10, actual_dr is 0.10.
    # pd_ratio = 0.10 / 0.10 = 1.0 → "pass".
    assert ratios["2016Q1"] == 1.0
    assert flags["2016Q1"] == "pass"
