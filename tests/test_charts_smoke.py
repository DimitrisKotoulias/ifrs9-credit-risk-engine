"""Smoke tests for the new reporting charts — verify each renders a PNG without error."""

import numpy as np
import pandas as pd

from credit_risk.reporting import charts


def test_plot_loss_distribution(tmp_path):
    losses = np.random.default_rng(0).uniform(0, 5e6, 10_000)
    measures = {"expected_loss": 1e6, "var": 3e6, "es": 3.5e6, "alpha": 0.999}
    charts.plot_loss_distribution(losses, measures, tmp_path)
    assert (tmp_path / "loss_distribution.png").exists()


def test_plot_km_survival(tmp_path):
    idx = np.arange(0, 60)
    curves = {
        "A": pd.DataFrame({"A": np.linspace(1.0, 0.9, 60)}, index=idx),
        "G": pd.DataFrame({"G": np.linspace(1.0, 0.5, 60)}, index=idx),
    }
    charts.plot_km_survival(curves, tmp_path)
    assert (tmp_path / "km_survival_curves.png").exists()


def test_plot_lgd_calibration(tmp_path):
    rng = np.random.default_rng(1)
    actual = rng.uniform(0, 1, 500)
    pred = np.clip(actual + rng.normal(0, 0.1, 500), 0, 1)
    decile = pd.DataFrame({
        "decile": range(10),
        "mean_predicted": np.linspace(0.1, 0.9, 10),
        "mean_actual": np.linspace(0.12, 0.88, 10),
        "count": [50] * 10,
    })
    charts.plot_lgd_calibration(actual, pred, decile, tmp_path)
    assert (tmp_path / "lgd_calibration.png").exists()


def test_plot_shock_tornado(tmp_path):
    df = pd.DataFrame({
        "scenario": ["PD +20%", "GFC-like", "LGD +10pp"],
        "delta_ecl": [1.2e6, 5.0e6, 2.0e6],
    })
    charts.plot_shock_tornado(df, tmp_path)
    assert (tmp_path / "ecl_shock_tornado.png").exists()


def test_plot_concentration(tmp_path):
    grouped = {
        "grade": pd.Series({"A": 100.0, "B": 200.0, "C": 50.0}),
        "purpose": pd.Series({"debt": 300.0, "car": 40.0}),
    }
    charts.plot_concentration(grouped, tmp_path)
    assert (tmp_path / "concentration_risk.png").exists()


def test_plot_pit_vs_ttc(tmp_path):
    pit = {
        "quarters": [f"20{y}Q{q}" for y in range(10, 14) for q in range(1, 5)],
        "default_rates": list(np.linspace(0.02, 0.08, 16)),
        "z_factors": list(np.linspace(1.0, -1.0, 16)),
        "ttc_pd": 0.05,
    }
    charts.plot_pit_vs_ttc(pit, tmp_path)
    assert (tmp_path / "pit_vs_ttc.png").exists()


def test_plot_calibration_by_vintage(tmp_path):
    df = pd.DataFrame({
        "group": ["2007-2012", "2013-2015", "2016-2018"],
        "pd_ratio_raw": [1.02, 0.95, 0.65],
        "pd_ratio_isotonic": [1.0, 0.99, 0.92],
    })
    charts.plot_calibration_by_vintage(df, tmp_path)
    assert (tmp_path / "calibration_by_vintage.png").exists()


def test_plot_shap_comparison(tmp_path):
    full = pd.DataFrame({
        "feature": ["int_rate", "grade", "dti", "fico"],
        "mean_abs_shap": [0.5, 0.4, 0.2, 0.1],
    })
    bureau = pd.DataFrame({
        "feature": ["dti", "fico", "revol_util"],
        "mean_abs_shap": [0.3, 0.25, 0.2],
    })
    charts.plot_shap_comparison(full, bureau, tmp_path)
    assert (tmp_path / "shap_comparison.png").exists()
