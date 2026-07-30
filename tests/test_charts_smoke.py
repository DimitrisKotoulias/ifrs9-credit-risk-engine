"""Tests for the reporting charts.

Every test here used to end at ``assert path.exists()``, which is why a chart could show
bars summing to 180% of the portfolio, a gains curve below the random diagonal, and a
literal backslash on an axis label without anything failing. File creation is the weakest
possible assertion about a figure; these tests inspect what was actually drawn.
"""

import numpy as np
import pandas as pd
import pytest

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
    fig = charts.plot_shock_tornado(df, tmp_path)
    assert (tmp_path / "ecl_shock_tornado.png").exists()
    # No LaTeX escapes on a matplotlib axis: `text.usetex` is off, so `\$` renders as a
    # literal backslash in the published figure.
    assert "\\" not in fig.axes[0].get_xlabel()


def test_plot_concentration(tmp_path):
    grouped = {
        "grade": pd.Series({"A": 100.0, "B": 200.0, "C": 50.0}),
        "purpose": pd.Series({"debt": 300.0, "car": 40.0}),
    }
    fig = charts.plot_concentration(grouped, tmp_path)
    assert (tmp_path / "concentration_risk.png").exists()
    # Untruncated dimensions must sum to exactly 100% of the portfolio.
    for ax in fig.axes:
        total = sum(p.get_width() for p in ax.patches)
        assert total == pytest.approx(100.0, abs=1e-6)


def test_plot_concentration_percentages_are_of_the_whole_portfolio(tmp_path):
    """With more categories than ``top_n``, the bars must not be rescaled to the top-N.

    The denominator was computed after ``nlargest(top_n)``, so every bar was inflated by
    portfolio-total / top-N-total while the axis read "% of Portfolio Exposure" --- and
    the figure then disagreed with the HHI table beside it. `addr_state` (50+ categories)
    triggered this on every production run, but the only test passed 3 categories, so the
    truncation branch never executed.
    """
    n_cat = 40
    exposures = pd.Series(
        {f"S{i:02d}": float(1000 - 10 * i) for i in range(n_cat)}
    ).sort_values(ascending=False)
    fig = charts.plot_concentration({"addr_state": exposures}, tmp_path, top_n=15)

    ax = fig.axes[0]
    assert len(ax.patches) == 15, "truncation branch did not run"
    shown = sum(p.get_width() for p in ax.patches)
    expected = float(exposures.nlargest(15).sum() / exposures.sum() * 100.0)
    assert shown == pytest.approx(expected, abs=1e-6)
    assert shown < 100.0, "top-15 of 40 categories cannot be 100% of the portfolio"
    assert "top 15 of 40" in ax.get_title()


def test_plot_ecl_tornado_marks_z0_as_anchor_not_baseline(tmp_path):
    """The Z=0 row is the unconditional anchor; the priced baseline has its own Z.

    This figure and ``plot_cutoff_profit`` --- the two charts behind the macro sensitivity
    section and Section 9 --- had no test at all, while a test existed for
    ``plot_shap_comparison``, which nothing in the pipeline calls.
    """
    df = pd.DataFrame({
        "macro_shock": [2.0, 0.0, -0.19, -2.0],
        "total_ecl": [0.9e9, 1.0e9, 1.02e9, 1.3e9],
        "coverage_ratio": [0.2, 0.22, 0.23, 0.3],
    })
    fig = charts.plot_ecl_tornado(
        df, tmp_path, scenario_shocks={"baseline": -0.19}
    )
    assert (tmp_path / "ecl_tornado.png").exists()

    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("unconditional anchor" in lbl for lbl in labels)
    assert any("baseline scenario" in lbl for lbl in labels)
    assert not any(lbl.strip() == "Z = 0.0 (Baseline)" for lbl in labels)

    # Adverse Z (negative) must raise ECL, favourable Z must lower it.
    widths = {lbl: p.get_width() for lbl, p in zip(labels, fig.axes[0].patches)}
    adverse = next(v for k, v in widths.items() if k.startswith("Z = -2"))
    benign = next(v for k, v in widths.items() if k.startswith("Z = +2"))
    assert adverse > 0 > benign


def test_plot_cutoff_profit_marks_the_supplied_operating_cutoff(tmp_path):
    strategy = pd.DataFrame({
        "cutoff": [500, 520, 540, 560],
        "approval_rate": [0.95, 0.80, 0.50, 0.10],
        "bad_rate": [0.20, 0.16, 0.11, 0.05],
        "expected_profit": [100e6, 120e6, 90e6, 20e6],
        "raroc": [0.10, 0.19, 0.25, 0.40],
    })
    fig = charts.plot_cutoff_profit(strategy, tmp_path, opt_cutoff=520)
    assert (tmp_path / "cutoff_profit_curve.png").exists()

    ax_profit = fig.axes[0]
    # The marked cutoff is the one passed in, not the profit argmax fallback.
    assert any(
        line.get_xdata()[0] == 520 for line in ax_profit.lines if len(line.get_xdata()) == 1
    )
    assert "Optimal cutoff = 520" in "".join(t.get_text() for t in ax_profit.texts)


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
    # Retained for the ad-hoc helper; note it is not part of the pipeline (see the
    # module docstring in reporting/charts.py).
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
