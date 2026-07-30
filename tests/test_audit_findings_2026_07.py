"""Regressions for the July-2026 codebase review (AUDIT_FINDINGS.md).

Each test pins a defect that reached a published report and that nothing in the suite could
see. Finding IDs are the ones used in that document.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.target import TARGET_COL, define_target
from credit_risk.models.pd_scorecard import (
    PDScorecard,
    _add_interaction_features,
    _encode_categoricals,
)
from credit_risk.utils.config import TargetConfig


def _fitted(small_accepted: pd.DataFrame) -> tuple[PDScorecard, pd.DataFrame]:
    """Fit a permissive scorecard on the fixture and return it with its feature frame."""
    df = define_target(
        small_accepted,
        TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"]),
    )
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    sc = PDScorecard(min_iv=0.001, max_iv=0.95, max_vif=50.0)
    sc.fit(X, y)
    return sc, X


# ── A4: the points table must add up to the model's own score ────────────────────


def test_points_sum_reconciles_with_predict_score(small_accepted):
    """Sum of a loan's attribute points == predict_score() for that loan.

    The intercept term carried a plus sign: ``(-woe*beta + alpha/n)``. Summed over n
    features that gives ``F(alpha - sum(woe*beta)) + O`` while predict_score returns
    ``F(-alpha - sum(woe*beta)) + O`` -- a constant ``2*alpha*F`` gap, -92.97 points on the
    production model. The published Appendix A table therefore topped out at 521 points
    against an operating cutoff of 530: an auditor adding it up would conclude that no
    applicant is ever approved.
    """
    sc, X = _fitted(small_accepted)

    woe = sc._woe_transform(X)
    feats = sc.feature_names
    alpha = float(sc._logit_result.params["const"])
    n = len(feats)
    betas = np.array([float(sc._logit_result.params[f]) for f in feats])

    per_feature = -(woe[feats].to_numpy() * betas + alpha / n) * sc._factor + sc._offset / n
    total_points = per_feature.sum(axis=1)

    np.testing.assert_allclose(
        total_points, np.asarray(sc.predict_score(X), dtype=float), atol=1e-6
    )


def test_published_points_table_uses_the_same_formula(small_accepted):
    """Every row of the exported table must obey the documented points equation."""
    sc, _ = _fitted(small_accepted)
    tbl = sc.scorecard_table
    assert not tbl.empty

    alpha = float(sc._logit_result.params["const"])
    n = len(sc.feature_names)
    expected = -(tbl["woe"] * tbl["beta"] + alpha / n) * sc._factor + sc._offset / n
    np.testing.assert_allclose(tbl["points"].to_numpy(), expected.to_numpy(), atol=1e-9)


# ── A5 / ST5: train/serve parity ─────────────────────────────────────────────────


def test_scoring_rebuilds_interaction_features(small_accepted):
    """Scoring a raw frame must equal scoring a pre-engineered one.

    ``_woe_transform`` used to skip ``_add_interaction_features``, so an interaction
    feature that survived selection arrived at scoring as an all-NaN column, landed in a
    Missing bin with zero training observations (WoE = 0) and was silently neutralised.
    The deployed model ran on one fewer predictor than the report documented.
    """
    sc, X = _fitted(small_accepted)
    prepped = _encode_categoricals(_add_interaction_features(X, sc._interaction_stats))

    np.testing.assert_allclose(sc.predict_proba(X), sc.predict_proba(prepped), atol=1e-12)


def test_interaction_features_are_actually_built_at_serve_time(small_accepted):
    """The fixture must exercise the interaction path, or the test above proves nothing."""
    sc, X = _fitted(small_accepted)
    engineered = _add_interaction_features(X, sc._interaction_stats)
    assert "revol_util_x_new_acc" in engineered.columns
    assert "dti_fico_interaction" in engineered.columns
    assert engineered["revol_util_x_new_acc"].notna().any()


def test_scoring_is_independent_of_the_frame_it_is_handed(small_accepted):
    """A row's PD must not depend on which other rows are scored with it.

    The z-scored interaction used the median/std of whatever frame was passed, so the
    challenger saw a different transform on train, test and OOT -- data-dependent
    preprocessing that partly undoes the out-of-time split.
    """
    sc, X = _fitted(small_accepted)
    full = sc.predict_proba(X)
    subset = sc.predict_proba(X.iloc[:50])
    np.testing.assert_allclose(subset, full[:50], atol=1e-12)


def test_missing_model_feature_raises_rather_than_scoring_on_nan(small_accepted):
    """Silently reindexing an absent model feature to NaN is what hid the skew."""
    sc, X = _fitted(small_accepted)
    raw_selected = [f for f in sc.feature_names if f in X.columns]
    assert raw_selected, "expected at least one selected feature to be a raw column"

    with pytest.raises(ValueError, match="absent from the frame"):
        sc.predict_proba(X.drop(columns=[raw_selected[0]]))


# ── B1 / G5: the sign constraint must hold on the delivered model ────────────────


def test_sign_check_converges(small_accepted):
    """No surviving coefficient may violate the WoE sign constraint.

    The check ran once -- drop violators, refit, stop -- and the refit redistributed signal
    onto the survivors, flipping ``mo_sin_rcnt_tl`` to +0.0182. The report claimed all
    coefficients were negative on the page before printing that number.
    """
    sc, _ = _fitted(small_accepted)
    coefs = sc._logit_result.params[sc.feature_names]
    assert (coefs < 0).all(), f"positive coefficients survived: {coefs[coefs >= 0].to_dict()}"
    assert sc.selection_stages["sign_check_rounds"][-1]["dropped"] == []


# ── G4: the gains chart must sit above the diagonal ──────────────────────────────


def test_gains_curve_orders_highest_risk_first():
    """Cumulative capture at 20% of the population must beat random on a real signal."""
    from credit_risk.validation.discrimination import compute_decile_table

    rng = np.random.default_rng(7)
    n = 5_000
    pd_hat = rng.uniform(0.01, 0.5, n)
    y = (rng.random(n) < pd_hat).astype(int)

    tbl = compute_decile_table(y, pd_hat, score_is_pd=True)
    capture_at_20pct = float(tbl.loc[tbl["decile"] == 2, "cum_bad_rate"].iloc[0])
    assert capture_at_20pct > 0.20, (
        f"gains curve is below the random diagonal ({capture_at_20pct:.3f} <= 0.20): "
        "the population is being ordered safest-first"
    )
    assert float(tbl.loc[tbl["decile"] == 1, "lift"].iloc[0]) > 1.0


# ── G6: post-origination FICO must be excluded by policy, not by an IV heuristic ──


def test_last_fico_is_on_the_leakage_deny_list(small_accepted):
    """``last_fico_range_*`` is the classic LendingClub leak and was never denied."""
    from credit_risk.data.leakage import filter_origination_features
    from credit_risk.utils.config import load_config

    cfg = load_config()
    deny = set(cfg.leakage.deny_list)
    assert {"last_fico_range_low", "last_fico_range_high"} <= deny

    df = small_accepted.copy()
    df["last_fico_range_low"] = 650.0
    df["last_fico_range_high"] = 654.0
    out = filter_origination_features(df, cfg.leakage)
    assert "last_fico_range_low" not in out.columns
    assert "last_fico_range_high" not in out.columns


# ── G3: interest rate must not be filled on the percent scale ────────────────────


def test_survival_frame_fills_int_rate_on_the_fraction_scale(small_accepted):
    """``.fillna(12.0)`` on a fraction column wrote 1200% into the Cox design matrix."""
    from credit_risk.models.survival import build_survival_frame

    df = define_target(
        small_accepted,
        TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"]),
    )
    df.loc[df.index[:50], "int_rate"] = np.nan
    surv = build_survival_frame(df)
    assert surv["int_rate"].max() <= 1.0, (
        f"int_rate reached {surv['int_rate'].max():.2f} -- filled on the percent scale"
    )


# ── G2: Cox hazard ratios must be comparable across covariates ───────────────────


def test_cox_covariates_are_standardised(small_accepted):
    """A ridge penalty on raw covariates punishes small-scale variables disproportionately.

    ``int_rate`` (a 0.05-0.35 fraction) was shrunk to a ~1% hazard effect across its entire
    observed range while the report named it a dominant hazard multiplier.
    """
    from credit_risk.models.survival import SurvivalPDModel

    df = define_target(
        small_accepted,
        TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"]),
    )
    model = SurvivalPDModel(sample_size=10_000, seed=0).fit(df)
    summary = model.cox_summary()

    assert "sd" in summary.columns, "the report needs the scale to convert back to units"
    assert model._cov_std is not None and (model._cov_std > 0).all()
    # Per-SD coefficients live on one scale, so no covariate can be orders of magnitude
    # smaller than the others purely because of its units.
    coefs = summary["coef"].abs()
    assert coefs.max() / max(coefs.min(), 1e-9) < 1e3
