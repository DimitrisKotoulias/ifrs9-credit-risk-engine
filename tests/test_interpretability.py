"""Tests for PDP / ICE interpretability plots."""

import numpy as np
import pandas as pd

from credit_risk.validation.interpretability import (
    partial_dependence_1d,
    plot_ice,
    plot_pdp_grid,
)


def _linear_predict(X: pd.DataFrame) -> np.ndarray:
    """Monotone-increasing probability in feature 'a'."""
    z = 0.1 * X["a"].to_numpy(dtype=float) - 0.05 * X["b"].to_numpy(dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.uniform(-5, 5, n), "b": rng.uniform(-5, 5, n)})


def test_partial_dependence_shapes():
    X = _frame()
    grid, pdp, ice = partial_dependence_1d(_linear_predict, X, "a", grid_size=20)
    assert grid.shape == (20,)
    assert pdp.shape == (20,)
    assert ice.shape == (len(X), 20)


def test_pdp_is_monotone_for_monotone_model():
    X = _frame(seed=1)
    _, pdp, _ = partial_dependence_1d(_linear_predict, X, "a", grid_size=25)
    # predict increases with 'a' => PDP should be non-decreasing.
    assert np.all(np.diff(pdp) >= -1e-9)


def test_pdp_bounded_probabilities():
    X = _frame(seed=2)
    _, pdp, ice = partial_dependence_1d(_linear_predict, X, "a")
    assert np.all((pdp >= 0.0) & (pdp <= 1.0))
    assert np.all((ice >= 0.0) & (ice <= 1.0))


def test_plot_pdp_grid_writes_file(tmp_path):
    X = _frame(seed=3)
    plot_pdp_grid(_linear_predict, X, ["a", "b"], tmp_path)
    assert (tmp_path / "pdp_grid.png").exists()


def test_plot_ice_writes_file(tmp_path):
    X = _frame(seed=4)
    plot_ice(_linear_predict, X, "a", tmp_path, n_ice=50)
    assert (tmp_path / "ice_plot.png").exists()
