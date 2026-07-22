"""Tests for the hazard-model lifetime PD calibration diagnostic.

This validates that the lifetime PD driving IFRS 9 ECL (hazard model term structure,
never passed through the scorecard's OOS recalibrator) is itself checked against
realised lifetime default rates by vintage. See lifetime_pd_calibration_by_vintage
in src/credit_risk/validation/calibration.py.
"""

import numpy as np

from credit_risk.validation.calibration import lifetime_pd_calibration_by_vintage


def test_perfect_calibration_in_band():
    """Predicted lifetime PD == realised default rate per vintage => ratio ~1, in_band."""
    rng = np.random.default_rng(0)
    n = 4000
    year = rng.integers(2008, 2017, n)  # all <= 2016 (matured)
    true_pd = 0.15
    y = (rng.uniform(0, 1, n) < true_pd).astype(float)
    pred = np.full(n, true_pd)

    result = lifetime_pd_calibration_by_vintage(y, pred, year)
    port = result["portfolio"]
    assert port["n"] == n
    assert abs(port["ratio"] - 1.0) < 0.15
    assert port["in_band"] is True
    assert len(result["by_vintage"]) == len(np.unique(year))


def test_maturity_filter_drops_immature_vintages():
    """Vintages after max_mature_year must be excluded from both rows and portfolio."""
    rng = np.random.default_rng(1)
    n_mature, n_immature = 2000, 2000
    year_mature = rng.integers(2008, 2017, n_mature)       # <= 2016
    year_immature = np.full(n_immature, 2018)              # excluded by default
    year = np.concatenate([year_mature, year_immature])
    pred = np.full(len(year), 0.10)
    # Immature vintage has NOT yet resolved: observed default rate artificially low.
    y_mature = (rng.uniform(0, 1, n_mature) < 0.10).astype(float)
    y_immature = np.zeros(n_immature)
    y = np.concatenate([y_mature, y_immature])

    result = lifetime_pd_calibration_by_vintage(y, pred, year, max_mature_year=2016)

    assert all(row["vintage_year"] <= 2016 for row in result["by_vintage"])
    assert result["portfolio"]["n"] == n_mature


def test_empty_input_guard():
    """No matured vintages present => n=0, ratio is NaN, in_band False, no crash."""
    year = np.full(100, 2018)
    y = np.zeros(100)
    pred = np.full(100, 0.10)

    result = lifetime_pd_calibration_by_vintage(y, pred, year, max_mature_year=2016)

    assert result["by_vintage"] == []
    assert result["portfolio"]["n"] == 0
    assert result["portfolio"]["in_band"] is False
    assert np.isnan(result["portfolio"]["ratio"])


def test_material_miscalibration_flagged_out_of_band():
    """Predicted lifetime PD several times the observed rate => out of the 0.5-1.5 band."""
    rng = np.random.default_rng(2)
    n = 3000
    year = rng.integers(2010, 2017, n)
    observed_rate = 0.05
    y = (rng.uniform(0, 1, n) < observed_rate).astype(float)
    pred = np.full(n, 0.30)  # 6x over-prediction

    result = lifetime_pd_calibration_by_vintage(y, pred, year)
    port = result["portfolio"]
    assert port["ratio"] > 1.5
    assert port["in_band"] is False
