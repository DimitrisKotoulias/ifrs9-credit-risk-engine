"""Tests for concentration risk (HHI + Granularity Adjustment)."""

import numpy as np
import pandas as pd

from credit_risk.risk.concentration import (
    effective_n,
    granularity_adjustment,
    herfindahl_index,
    hhi_by_dimension,
    run_concentration,
)


def test_hhi_single_name_equals_one():
    assert herfindahl_index(np.array([1000.0])) == 1.0


def test_hhi_uniform_equals_one_over_n():
    n = 8
    assert abs(herfindahl_index(np.ones(n)) - 1.0 / n) < 1e-12


def test_hhi_bounds():
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.uniform(1, 1000, rng.integers(2, 50))
        hhi = herfindahl_index(e)
        assert 0.0 < hhi <= 1.0


def test_hhi_ignores_zero_and_negative():
    assert abs(herfindahl_index(np.array([10.0, 10.0, 0.0, -5.0])) - 0.5) < 1e-12


def test_effective_n_inverse_of_hhi():
    assert abs(effective_n(0.25) - 4.0) < 1e-12


def test_granularity_adjustment_non_negative_and_shrinks_with_granularity():
    rng = np.random.default_rng(1)
    pd_arr = rng.uniform(0.01, 0.2, 500)
    lgd = rng.uniform(0.3, 0.6, 500)
    # Concentrated: one giant exposure. Granular: many equal small ones.
    ead_conc = np.concatenate([[1e7], np.full(499, 1.0)])
    ead_gran = np.full(500, 1e7 / 500)
    ga_conc = granularity_adjustment(pd_arr, lgd, ead_conc)
    ga_gran = granularity_adjustment(pd_arr, lgd, ead_gran)
    assert ga_conc >= 0.0 and ga_gran >= 0.0
    assert ga_conc > ga_gran


def test_hhi_by_dimension_frame():
    df = pd.DataFrame({
        "grade": ["A", "A", "B", "C", "C", "C"],
        "funded_amnt": [1000.0, 1000.0, 2000.0, 500.0, 500.0, 500.0],
    })
    out = hhi_by_dimension(df, ["grade"], "funded_amnt")
    assert list(out["dimension"]) == ["grade"]
    assert 0.0 < float(out.iloc[0]["hhi"]) <= 1.0
    assert int(out.iloc[0]["n_categories"]) == 3


def test_run_concentration_summary_and_grouped():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame({
        "grade": rng.choice(list("ABCDEFG"), n),
        "purpose": rng.choice(["debt", "car", "home"], n),
        "addr_state": rng.choice(["CA", "NY", "TX", "FL"], n),
        "funded_amnt": rng.uniform(5_000, 30_000, n),
        "pd_pred": rng.uniform(0.01, 0.2, n),
        "lgd_pred": rng.uniform(0.3, 0.6, n),
        "ead": rng.uniform(5_000, 30_000, n),
    })
    summary, grouped = run_concentration(df)
    assert "dimensions" in summary and "granularity_adjustment" in summary
    assert len(summary["dimensions"]) == 3
    assert set(grouped) == {"grade", "purpose", "addr_state"}
    assert summary["granularity_adjustment"] >= 0.0
