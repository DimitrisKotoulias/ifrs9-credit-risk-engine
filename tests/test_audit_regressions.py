"""Regression tests for the defects recorded in the internal audit log.

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


# ── A1 follow-up: the recalibration gate must look at out-of-time evidence ───────


def _miscalibrated_oot(n=4000, seed=0):
    """Well-ranked but systematically under-predicting scores, as in the 2016-18 drift."""
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0.02, 0.45, size=n)
    # Realised default rate runs ~1.5x the prediction: the era drift, in miniature.
    y = (rng.uniform(size=n) < np.clip(p_raw * 1.5, 0, 1)).astype(int)
    return y, p_raw


def test_gate_triggers_on_out_of_time_miscalibration():
    from credit_risk.validation.calibration import select_oot_recalibrator

    y, p = _miscalibrated_oot()
    half = len(y) // 2
    gate = select_oot_recalibrator(y[:half], p[:half], y[half:], p[half:])

    assert gate["triggered"], "systematic under-prediction must trigger the gate"
    assert "ratio" in gate["trigger_reason"] or "Hosmer" in gate["trigger_reason"]


def test_gate_does_not_trigger_when_already_calibrated():
    from credit_risk.validation.calibration import select_oot_recalibrator

    rng = np.random.default_rng(1)
    n = 4000
    p = rng.uniform(0.05, 0.40, size=n)
    y = (rng.uniform(size=n) < p).astype(int)  # honestly calibrated
    half = n // 2
    gate = select_oot_recalibrator(y[:half], p[:half], y[half:], p[half:])

    assert not gate["triggered"]
    assert gate["calibrator"] is None
    assert gate["chosen_method"] == "none"


def test_gate_accepts_only_on_improvement_in_the_held_out_slice():
    from credit_risk.validation.calibration import select_oot_recalibrator

    y, p = _miscalibrated_oot(seed=2)
    half = len(y) // 2
    gate = select_oot_recalibrator(y[:half], p[:half], y[half:], p[half:])

    assert gate["triggered"]
    if gate["accepted"]:
        before = abs(gate["eval_before"]["ratio"] - 1.0)
        after = abs(gate["eval_after"]["ratio"] - 1.0)
        assert after < before, "acceptance requires the ratio to move toward 1.0"
        assert gate["calibrator"] is not None
    else:
        assert gate["calibrator"] is None, "a rejected transform must not be attached"


def test_gate_is_fitted_on_the_earlier_slice_only():
    """The evaluation slice must never be used for fitting.

    Guards the property that made the old design defensible and the new one honest: a
    transform fitted on the reporting data would trivially pass any calibration test.
    """
    from credit_risk.validation.calibration import select_oot_recalibrator

    y, p = _miscalibrated_oot(seed=3)
    half = len(y) // 2
    gate = select_oot_recalibrator(y[:half], p[:half], y[half:], p[half:])
    if gate["calibrator"] is None:
        pytest.skip("gate rejected the transform on this draw")

    # Perfect in-sample calibration on the eval slice would betray a leak.
    from credit_risk.validation.calibration import _apply_calibrator

    ratio = float(np.mean(_apply_calibrator(gate["calibrator"], p[half:])) / np.mean(y[half:]))
    assert abs(ratio - 1.0) > 1e-9, "exact eval-slice calibration implies the fit leaked"


def test_chronological_split_keeps_the_earlier_half():
    from credit_risk.validation.calibration import chronological_oot_split

    dates = pd.to_datetime(pd.Series(["2016-01-01"] * 10 + ["2018-01-01"] * 10))
    mask = chronological_oot_split(dates.to_numpy(), fit_fraction=0.5)
    assert mask[:10].all() and not mask[10:].any()


def test_gate_skips_when_slices_are_too_small():
    from credit_risk.validation.calibration import select_oot_recalibrator

    y = np.array([0, 1] * 20)
    p = np.linspace(0.1, 0.5, 40)
    gate = select_oot_recalibrator(y[:20], p[:20], y[20:], p[20:])
    assert gate["calibrator"] is None
    assert "too small" in gate.get("skip_reason", "")


# ── C5: macro scenario axes must actually separate upside from downside ─────────


def test_no_macro_scenario_axis_is_degenerate():
    """Every macro axis must differ between upside and downside after flooring.

    FEDFUNDS previously floored to 0.1 in both, so the policy-rate channel contributed
    nothing to scenario separation, and CPI_inflation had no delta at all.
    """
    from credit_risk.risk.ifrs9_ecl import _SCENARIO_DELTAS

    base = {
        "UNRATE": 7.633, "GDP_growth": 0.739, "FEDFUNDS": 0.501,
        "CPI_inflation": 0.425, "HPI_growth": -0.119,
    }
    up = {k: base[k] + _SCENARIO_DELTAS["upside"].get(k, 0.0) for k in base}
    down = {k: base[k] + _SCENARIO_DELTAS["downside"].get(k, 0.0) for k in base}
    up["FEDFUNDS"] = max(0.1, up["FEDFUNDS"])
    down["FEDFUNDS"] = max(0.1, down["FEDFUNDS"])
    up["UNRATE"] = max(2.0, up["UNRATE"])
    up["CPI_inflation"] = max(0.0, up["CPI_inflation"])

    identical = [k for k in base if abs(up[k] - down[k]) < 1e-9]
    assert not identical, f"axes identical in both scenarios: {identical}"


def test_every_scenario_delta_agrees_with_its_sign_prior():
    """Ordering must hold by construction, not by coefficient luck.

    Projection coefficients carry imposed economic signs. If any axis moves the "wrong"
    way in a scenario, that axis partially offsets the scenario, and whether
    Downside >= Baseline >= Upside still holds then depends on the fitted magnitudes --
    a different sample could silently invert it. Requiring every axis to agree with its
    prior makes the ordering structural.
    """
    from credit_risk.risk.ifrs9_ecl import _MACRO_SIGN_PRIORS, _SCENARIO_DELTAS

    up, down = _SCENARIO_DELTAS["upside"], _SCENARIO_DELTAS["downside"]
    for axis, prior in _MACRO_SIGN_PRIORS.items():
        # prior * delta > 0 means the axis pushes defaults up.
        assert prior * down[axis] > 0, f"downside {axis} does not raise defaults"
        assert prior * up[axis] < 0, f"upside {axis} does not lower defaults"


def test_scenario_ordering_is_independent_of_coefficient_magnitudes():
    """With every axis aligned to its prior, any non-negative magnitudes preserve order."""
    from credit_risk.risk.ifrs9_ecl import _MACRO_SIGN_PRIORS, _SCENARIO_DELTAS

    rng = np.random.default_rng(7)
    for _ in range(200):
        mags = {k: float(rng.uniform(0.0, 0.05)) for k in _MACRO_SIGN_PRIORS}
        coefs = {k: _MACRO_SIGN_PRIORS[k] * mags[k] for k in mags}
        dr_up = sum(coefs[k] * _SCENARIO_DELTAS["upside"][k] for k in coefs)
        dr_down = sum(coefs[k] * _SCENARIO_DELTAS["downside"][k] for k in coefs)
        assert dr_down >= 0.0 >= dr_up, "scenario ordering inverted for some magnitudes"
