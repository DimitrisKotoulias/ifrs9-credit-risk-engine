"""Regression tests for the defects recorded in docs/AUDIT.md.

Each test pins a behaviour that was previously wrong. See the audit document for the
evidence and impact of each finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.models.lgd import LGDModel
from credit_risk.models.pd_challenger import PDChallenger, _resolve_features
from credit_risk.risk.ifrs9_ecl import (
    SICRConfig,
    assign_stages,
    compute_ecl_single_scenario,
    term_horizon_mask,
)


# ── B1: lifetime ECL must stop at each loan's own contractual term ───────────────


def test_term_horizon_mask_truncates_at_each_loan_term():
    mask = term_horizon_mask(np.array([36.0, 60.0]), n=2, horizon=60)
    assert mask is not None
    assert mask[0].sum() == 36, "36-month loan should expose exactly 36 months"
    assert mask[1].sum() == 60, "60-month loan should expose all 60 months"
    assert not mask[0, 36:].any(), "months beyond the term must be masked out"


def test_lifetime_ecl_ignores_months_beyond_the_loan_term():
    """A 36-month loan must not be charged for defaults in months 37-60.

    The term-structure matrix is sized at the portfolio maximum term, so without the
    per-loan mask a 36-month loan accrued 60 months of marginal PD into its lifetime ECL.
    """
    horizon = 60
    marginal_pd = np.full((1, horizon), 0.001)
    lgd = np.array([0.9])
    ead = np.array([10_000.0])
    eir = np.array([0.0])  # no discounting, so the sum is exactly PD * LGD * EAD
    stages = np.array([2])  # Stage 2 => lifetime ECL

    unmasked = compute_ecl_single_scenario(marginal_pd, lgd, ead, eir, stages)
    masked = compute_ecl_single_scenario(
        marginal_pd, lgd, ead, eir, stages, term_months=np.array([36.0])
    )

    assert unmasked[0] == pytest.approx(60 * 0.001 * 0.9 * 10_000.0)
    assert masked[0] == pytest.approx(36 * 0.001 * 0.9 * 10_000.0)
    assert masked[0] < unmasked[0]


def test_full_term_loan_is_unaffected_by_the_mask():
    horizon = 60
    marginal_pd = np.full((1, horizon), 0.001)
    args = (marginal_pd, np.array([0.9]), np.array([10_000.0]), np.array([0.0]), np.array([2]))
    assert compute_ecl_single_scenario(*args, term_months=np.array([60.0])) == pytest.approx(
        compute_ecl_single_scenario(*args)
    )


# ── B2: the challenger must not silently train on fewer features ─────────────────


def test_resolve_features_raises_on_missing_column():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="absent from the training frame"):
        _resolve_features(["a", "b", "grade_enc"], X, "PDChallenger")


def test_resolve_features_still_drops_target_and_date_columns():
    X = pd.DataFrame({"a": [1.0], "target": [0], "issue_d": ["Jan-2015"]})
    assert _resolve_features(["a", "target", "issue_d"], X, "PDChallenger") == ["a"]


def test_challenger_fit_refuses_a_reduced_feature_set():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] > 0).astype(int))
    with pytest.raises(ValueError, match="absent from the training frame"):
        PDChallenger(seed=0).fit(X, y, X, y, feature_names=["a", "b", "grade_enc"])


# ── B3: downturn LGD comes from realised severity, not from fitted values ────────


def test_downturn_lgd_reflects_realised_severity_not_shrunk_predictions():
    """Realised severity is bimodal; predictions collapse toward the mean.

    Taking the p90 over predictions produced a downturn LGD barely above mean LGD. It must
    instead track the realised severity distribution, which has mass near total loss.
    """
    model = LGDModel(downturn_percentile=90.0)
    realised = pd.Series([0.05, 0.10, 0.95, 0.98, 1.00, 0.99, 0.97, 0.96, 0.94, 1.00])
    predictions = pd.Series(np.full(len(realised), 0.70))  # fully shrunk to the mean

    downturn = model._downturn_from_realised(realised[realised > 0], predictions, realised)

    assert downturn == pytest.approx(float(np.percentile(realised, 90.0)))
    assert downturn > predictions.mean(), "downturn must exceed the central estimate"
    # The old implementation would have returned the p90 of a constant vector: 0.70.
    assert downturn > 0.70


def test_downturn_lgd_never_falls_below_mean_predicted_lgd():
    model = LGDModel(downturn_percentile=90.0)
    realised = pd.Series([0.10, 0.12, 0.11, 0.13])
    predictions = pd.Series(np.full(4, 0.60))
    assert model._downturn_from_realised(realised, predictions, realised) == pytest.approx(0.60)


# ── B4: the relative SICR trigger must be skipped, not faked ─────────────────────


def _staging_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"target": np.zeros(n, dtype=int)})


def test_absolute_sicr_still_fires_without_an_origination_pd():
    """Passing None must not disable Stage 2 wholesale.

    Previously the entire Stage 2 block was gated on `pd_orig_lifetime is not None`, so
    omitting the origination snapshot silently produced a two-stage portfolio.
    """
    cfg = SICRConfig(pd_multiplier=2.5, abs_threshold=0.20, dpd_backstop=30)
    df = _staging_frame(3)
    pd_lifetime = np.array([0.05, 0.25, 0.50])

    stages = assign_stages(df, pd_lifetime, None, cfg)

    assert stages.tolist() == [1, 2, 2]


def test_relative_sicr_fires_only_with_a_genuine_origination_pd():
    cfg = SICRConfig(pd_multiplier=2.5, abs_threshold=0.90, dpd_backstop=30)
    df = _staging_frame(2)
    pd_lifetime = np.array([0.30, 0.05])
    pd_orig = np.array([0.05, 0.04])  # first loan tripled; second barely moved

    assert assign_stages(df, pd_lifetime, None, cfg).tolist() == [1, 1]
    assert assign_stages(df, pd_lifetime, pd_orig, cfg).tolist() == [2, 1]


def test_defaulted_loans_are_stage_3_regardless_of_sicr_inputs():
    cfg = SICRConfig()
    df = pd.DataFrame({"target": [1, 0]})
    stages = assign_stages(df, np.array([0.01, 0.01]), None, cfg)
    assert stages.tolist() == [3, 1]
