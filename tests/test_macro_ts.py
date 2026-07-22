"""Tests for the macro time-series diagnostics (ADF / Granger / AIC / VECM)."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.validation.macro_ts import analyze_macro_timeseries

pytest.importorskip("statsmodels")


def _synthetic_macro(n: int = 44, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # UNRATE as a mean-reverting series; default rate driven positively by UNRATE.
    unrate = np.zeros(n)
    unrate[0] = 6.0
    for t in range(1, n):
        unrate[t] = 0.8 * unrate[t - 1] + 0.2 * 6.0 + rng.normal(0, 0.4)
    gdp = rng.normal(2.0, 1.0, n)
    default_rate = 0.02 + 0.010 * (unrate - 6.0) - 0.002 * (gdp - 2.0) + rng.normal(0, 0.002, n)
    quarters = pd.period_range("2008Q1", periods=n, freq="Q").astype(str)
    return pd.DataFrame({
        "quarter": quarters,
        "default_rate": np.clip(default_rate, 0.001, None),
        "UNRATE": unrate,
        "GDP_growth": gdp,
        "FEDFUNDS": rng.uniform(0.1, 2.0, n),
        "CPI_inflation": rng.uniform(1.0, 3.0, n),
    })


def test_analyze_returns_expected_keys():
    res = analyze_macro_timeseries(_synthetic_macro(), max_lag=4)
    assert set(res) == {"n_quarters", "adf", "granger", "aic_lag_selection", "johansen"}
    assert res["n_quarters"] == 44


def test_aic_recovers_positive_unrate_sign():
    res = analyze_macro_timeseries(_synthetic_macro(seed=1), max_lag=4)
    aic = res["aic_lag_selection"]
    assert aic is not None
    # The data-generating process has a positive UNRATE -> default relationship.
    assert aic["unrate_coef"] > 0
    assert aic["unrate_sign_ok"] is True


def test_adf_present_for_default_rate():
    res = analyze_macro_timeseries(_synthetic_macro(seed=2), max_lag=4)
    assert "default_rate" in res["adf"]


def test_short_series_degrades_gracefully():
    short = _synthetic_macro(n=5, seed=3)
    res = analyze_macro_timeseries(short, max_lag=4)
    # Must not raise; sub-analyses that cannot run return None.
    assert res["n_quarters"] == 5
    assert res["granger"] is None
    assert res["johansen"] is None


def test_missing_unrate_column_handled():
    df = _synthetic_macro(seed=4).drop(columns=["UNRATE"])
    res = analyze_macro_timeseries(df, max_lag=4)
    assert res["granger"] is None
