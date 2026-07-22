"""Tests for the paired bootstrap A/B test (champion vs challenger)."""

import numpy as np
import pytest

from credit_risk.validation.ab_test import paired_bootstrap_gini


def _labels_and_scores(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # A weak signal; B a stronger signal correlated with y.
    a = 0.5 * y + rng.normal(0, 1.0, n)
    b = 1.5 * y + rng.normal(0, 1.0, n)
    return y, a, b


def test_returns_expected_structure():
    y, a, b = _labels_and_scores()
    res = paired_bootstrap_gini(y, a, b, n_boot=500, seed=1)
    for key in ("gini_a", "gini_b", "diff"):
        assert set(res[key]) == {"median", "lo", "hi"}
    assert "significant" in res and "n_boot_valid" in res
    assert res["n_boot_valid"] > 0


def test_stronger_model_has_positive_significant_diff():
    y, a, b = _labels_and_scores(n=2000, seed=2)
    res = paired_bootstrap_gini(y, a, b, n_boot=800, seed=2)
    assert res["diff"]["median"] > 0
    assert res["significant"] is True  # B clearly better => CI excludes 0


def test_identical_models_diff_contains_zero():
    y, a, _ = _labels_and_scores(n=1500, seed=3)
    res = paired_bootstrap_gini(y, a, a.copy(), n_boot=800, seed=3)
    assert abs(res["diff"]["median"]) < 1e-9
    assert res["diff"]["lo"] <= 0.0 <= res["diff"]["hi"]
    assert res["significant"] is False


def test_ci_ordering():
    y, a, b = _labels_and_scores(seed=4)
    res = paired_bootstrap_gini(y, a, b, n_boot=500, seed=4)
    for key in ("gini_a", "gini_b", "diff"):
        s = res[key]
        assert s["lo"] <= s["median"] <= s["hi"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_gini(np.array([0, 1]), np.array([0.1, 0.2]), np.array([0.3]))
