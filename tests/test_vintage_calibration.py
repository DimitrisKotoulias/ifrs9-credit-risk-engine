"""Tests for era-specific vintage calibration."""

import numpy as np

from credit_risk.validation.calibration import (
    calibration_by_vintage_group,
    fit_era_calibrators,
)


def _drifted_data(seed: int = 0):
    """Well-calibrated early era; late era under-predicts (pred ~ half of actual)."""
    rng = np.random.default_rng(seed)
    n = 6000
    year = rng.integers(2008, 2019, n)
    late = year >= 2016
    # Base predicted PD.
    pred = rng.uniform(0.02, 0.30, n)
    # Actual default prob: early matches pred; late has ~2x the pred (under-prediction).
    p_true = np.where(late, np.clip(pred * 2.0, 0, 1), pred)
    y = (rng.uniform(0, 1, n) < p_true).astype(float)
    return y, pred, year


def test_fit_era_calibrators_two_eras():
    y, pred, year = _drifted_data()
    cals = fit_era_calibrators(y, pred, year, split_year=2016)
    assert set(cals) == {"early", "late"}
    assert "isotonic" in cals["early"] and "platt" in cals["late"]


def test_vintage_group_table_structure():
    y, pred, year = _drifted_data(seed=1)
    df = calibration_by_vintage_group(y, pred, year, split_year=2016)
    expected = {"group", "n", "raw_pd", "isotonic_pd", "platt_pd", "actual_dr",
                "pd_ratio_raw", "pd_ratio_isotonic"}
    assert expected.issubset(df.columns)
    assert len(df) >= 2


def test_recalibration_moves_ratio_toward_one():
    """For the drifted late era, the isotonic ratio should be closer to 1 than raw."""
    y, pred, year = _drifted_data(seed=2)
    df = calibration_by_vintage_group(y, pred, year, split_year=2016)
    late = df[df["group"] == "2016-2018"].iloc[0]
    assert abs(late["pd_ratio_isotonic"] - 1.0) < abs(late["pd_ratio_raw"] - 1.0)


def test_raw_ratio_below_one_for_underpredicted_era():
    y, pred, year = _drifted_data(seed=3)
    df = calibration_by_vintage_group(y, pred, year, split_year=2016)
    late = df[df["group"] == "2016-2018"].iloc[0]
    assert late["pd_ratio_raw"] < 1.0  # under-prediction
