"""Tests for the Monte Carlo economic-capital module (ASRF loss distribution)."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.risk.economic_capital import (
    risk_measures,
    run_economic_capital,
    simulate_portfolio_losses,
)


def _toy_portfolio(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pd_arr = rng.uniform(0.01, 0.20, n)
    lgd_arr = rng.uniform(0.30, 0.60, n)
    ead_arr = rng.uniform(5_000, 30_000, n)
    return pd_arr, lgd_arr, ead_arr


def test_risk_measures_ordering_es_ge_var_ge_el():
    pd_arr, lgd_arr, ead_arr = _toy_portfolio()
    losses = simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, n_sim=20_000, seed=1)
    m = risk_measures(losses, alpha=0.999)
    assert m["es"] >= m["var"] >= m["expected_loss"] > 0.0
    assert m["economic_capital"] >= m["unexpected_loss"] >= 0.0


def test_losses_non_negative():
    pd_arr, lgd_arr, ead_arr = _toy_portfolio()
    losses = simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, n_sim=5_000, seed=2)
    assert np.all(losses >= 0.0)


def test_seed_determinism():
    pd_arr, lgd_arr, ead_arr = _toy_portfolio()
    a = simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, n_sim=5_000, seed=7)
    b = simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, n_sim=5_000, seed=7)
    assert np.array_equal(a, b)


def test_expected_loss_matches_analytic():
    """Mean simulated loss should track the analytic EL = sum(PD*LGD*EAD)."""
    pd_arr, lgd_arr, ead_arr = _toy_portfolio(n=300, seed=3)
    analytic_el = float((pd_arr * lgd_arr * ead_arr).sum())
    losses = simulate_portfolio_losses(
        pd_arr, lgd_arr, ead_arr, n_sim=40_000, seed=3, n_buckets=50
    )
    el = float(losses.mean())
    assert abs(el - analytic_el) / analytic_el < 0.05


def test_higher_rho_fatter_tail():
    """Greater asset correlation concentrates systematic risk => larger ES."""
    pd_arr, lgd_arr, ead_arr = _toy_portfolio(n=400, seed=4)
    lo = simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, rho=0.05, n_sim=30_000, seed=4)
    hi = simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, rho=0.30, n_sim=30_000, seed=4)
    assert risk_measures(hi)["es"] > risk_measures(lo)["es"]


def test_rho_out_of_range_raises():
    pd_arr, lgd_arr, ead_arr = _toy_portfolio(n=10)
    with pytest.raises(ValueError):
        simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, rho=1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        simulate_portfolio_losses(np.array([0.1, 0.2]), np.array([0.4]), np.array([1.0]))


def test_run_economic_capital_dataframe():
    pd_arr, lgd_arr, ead_arr = _toy_portfolio(n=150, seed=5)
    df = pd.DataFrame({"pd_pred": pd_arr, "lgd_pred": lgd_arr, "ead": ead_arr})
    losses, measures = run_economic_capital(df, n_sim=10_000, seed=5)
    assert len(losses) == 10_000
    assert measures["es"] >= measures["var"] >= measures["expected_loss"]


def test_empty_portfolio_returns_zero_losses():
    losses = simulate_portfolio_losses(
        np.array([]), np.array([]), np.array([]), n_sim=1_000
    )
    assert losses.shape == (1_000,)
    assert np.all(losses == 0.0)
