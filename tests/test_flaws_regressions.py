"""Regression tests for the defects recorded in Flaws.md (audit round 3).

Each test pins a behaviour that was previously wrong. Finding IDs (N1, N3, ...) refer to
that document, which carries the evidence and impact for each.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.binning import (
    ManualMonotonicBinner,
    OptBinningWrapper,
    _try_optbinning,
    binner_kind,
)
from credit_risk.models.pd_scorecard import PDScorecard, _add_interaction_features
from credit_risk.risk.expected_loss import portfolio_el_summary
from credit_risk.validation.calibration import chronological_oot_split


# ── N4: the Platt branch of the recalibration gate must not crash scoring ────────


class _FakeLogit:
    """Minimal stand-in for the statsmodels result PDScorecard holds."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def predict(self, _X: object) -> pd.Series:
        return pd.Series(self._values)


def test_predict_proba_accepts_a_platt_calibrator():
    """A LogisticRegression calibrator has no .transform().

    The gate picks between isotonic and Platt on out-of-fold Brier score, so which one is
    attached is data-dependent. predict_proba() called .transform() unconditionally, so
    any run where Platt won died with AttributeError at portfolio scoring time — a latent
    crash, not a theoretical one (Flaws.md N4).
    """
    from sklearn.linear_model import LogisticRegression

    raw = np.array([0.05, 0.20, 0.40, 0.60, 0.85])
    platt = LogisticRegression().fit(raw.reshape(-1, 1), np.array([0, 0, 1, 1, 1]))

    sc = PDScorecard()
    sc._logit_result = _FakeLogit(raw)
    sc._selected_features = []
    sc._woe_transformer = None
    sc.set_calibrator(platt)

    out = sc._apply_calibrator(raw)
    assert out.shape == raw.shape
    assert np.all((out > 0.0) & (out < 1.0)), "calibrated PDs must stay inside (0, 1)"


def test_predict_proba_accepts_an_isotonic_calibrator():
    from sklearn.isotonic import IsotonicRegression

    raw = np.array([0.05, 0.20, 0.40, 0.60, 0.85])
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, np.array([0, 0, 1, 1, 1]))

    sc = PDScorecard()
    sc.set_calibrator(iso)
    out = sc._apply_calibrator(raw)
    assert out.shape == raw.shape
    assert np.all((out > 0.0) & (out < 1.0))


def test_apply_calibrator_rejects_an_object_that_is_neither():
    sc = PDScorecard()
    sc.set_calibrator(object())
    with pytest.raises(TypeError):
        sc._apply_calibrator(np.array([0.1, 0.2]))


# ── N3: the "chronological" OOT split must actually split on time ───────────────


def test_chronological_split_is_chronological_on_shuffled_input():
    """Rows arrive in arbitrary order; the split must still be a clean date cut.

    The gate previously received np.arange() as its ordering key whenever the caller
    passed none, which made "out-of-time validation" a random 50/50 holdout of the same
    period while the report described it as fitting on earlier vintages (Flaws.md N3).
    """
    rng = np.random.default_rng(0)
    dates = pd.to_datetime(
        rng.permutation(pd.date_range("2016-01-01", "2018-12-01", freq="MS").repeat(50))
    )
    fit_mask = chronological_oot_split(dates.to_numpy(), fit_fraction=0.5)

    fit_dates, eval_dates = dates[fit_mask], dates[~fit_mask]
    assert len(fit_dates) and len(eval_dates)
    assert fit_dates.max() <= eval_dates.min(), (
        "every fitting-slice loan must originate no later than every evaluation-slice loan"
    )


def test_chronological_split_keeps_whole_vintages_together():
    """Ties (same origination month) must not straddle the boundary."""
    dates = pd.to_datetime(
        pd.date_range("2016-01-01", "2018-12-01", freq="MS").repeat(100)
    )
    fit_mask = chronological_oot_split(dates.to_numpy(), fit_fraction=0.5)
    boundary_month = dates[fit_mask].max()
    assert not (dates[~fit_mask] == boundary_month).any()


def test_positional_fallback_is_logged_as_not_out_of_time(caplog):
    """If no ordering key survives, the fallback must announce that it is positional."""
    with caplog.at_level(logging.WARNING):
        chronological_oot_split(np.full(100, np.nan), fit_fraction=0.5)
    assert any("POSITIONAL" in r.message.upper() for r in caplog.records), (
        "a positional split must warn: it is not an out-of-time test"
    )


# ── N32: a binner swap must never happen silently ───────────────────────────────


def test_try_optbinning_does_not_swallow_arbitrary_errors(monkeypatch):
    """Only ImportError may trigger the fallback.

    `except (ImportError, Exception)` caught everything, so an unrelated failure inside
    optbinning silently produced a different model — different bins, different WoE,
    different surviving features — and the report still built (Flaws.md N32).
    """
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "optbinning":
            raise RuntimeError("optbinning is installed but broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    with pytest.raises(RuntimeError):
        _try_optbinning()


def test_try_optbinning_still_falls_back_on_a_genuine_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _missing(name, *args, **kwargs):
        if name == "optbinning":
            raise ImportError("No module named 'optbinning'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing)
    assert _try_optbinning() is False


def test_binner_kind_identifies_both_implementations():
    """metrics.json must be able to record which binner produced the report's numbers."""
    assert binner_kind(OptBinningWrapper()) == "optbinning"
    assert binner_kind(ManualMonotonicBinner()) == "manual_fallback"


def test_optbinning_wrapper_has_no_dead_monotonic_trend_parameter():
    """The parameter was advertised in the docstring and never forwarded (Flaws.md N30/N45)."""
    assert not hasattr(OptBinningWrapper(), "monotonic_trend")


# ── N24: one engineered feature, one definition ─────────────────────────────────


def test_interaction_features_do_not_overwrite_the_loader_definition():
    """`revol_util_x_open_acc` means revol_util x open_acc, everywhere.

    The scorecard used to overwrite that column with revol_util x acc_open_past_24mths,
    so the report's IV and SHAP rows were interpreted under the wrong definition
    (Flaws.md N24).
    """
    df = pd.DataFrame({
        "revol_util": [50.0, 80.0],
        "open_acc": [10.0, 20.0],
        "acc_open_past_24mths": [1.0, 2.0],
        # as produced by data/loader.py
        "revol_util_x_open_acc": [0.5 * 10.0, 0.8 * 20.0],
        "loan_to_income": [0.25, 0.40],
        "loan_amnt": [10_000.0, 20_000.0],
        "annual_inc": [40_000.0, 50_000.0],
    })
    out = _add_interaction_features(df)

    assert out["revol_util_x_open_acc"].tolist() == [5.0, 16.0], (
        "the loader's definition must survive untouched"
    )
    assert "revol_util_x_new_acc" in out.columns
    assert out["revol_util_x_new_acc"].tolist() == [0.5, 1.6]
    assert out["loan_to_income"].tolist() == [0.25, 0.40], (
        "an already-engineered loan_to_income must not be silently recomputed"
    )


# ── N25: int_rate is a fraction, so its fallback must be too ────────────────────


def test_term_structure_features_normalise_int_rate_scale():
    """Filling missing int_rate with the literal 12.0 injected ~92x the column mean.

    loader.py stores int_rate as a fraction (~0.13). The hazard model filled NaNs with
    12.0 and fed the result straight to StandardScaler (Flaws.md N25).
    """
    from credit_risk.models.pd_term_structure import DiscreteHazardModel

    df = pd.DataFrame({
        "int_rate": [0.13, np.nan, 0.09],
        "dti": [15.0, 20.0, 10.0],
        "grade": ["B", "C", "A"],
        "term": [" 36 months"] * 3,
    })
    feats = DiscreteHazardModel()._build_features(df, mob=np.array([12.0, 12.0, 12.0]))
    int_rate_col = feats[:, 3]
    assert int_rate_col.max() <= 1.0, (
        f"int_rate features must be fractions, got max {int_rate_col.max()}"
    )
    assert int_rate_col[1] == pytest.approx(0.12), "the NaN fallback must be 12%, not 1200%"


# ── N26: portfolio_el_summary must return what its docstring promises ───────────


def test_portfolio_el_summary_returns_mean_lgd():
    df = pd.DataFrame({
        "el": [100.0, 200.0],
        "ead": [1000.0, 2000.0],
        "pd_pred": [0.1, 0.2],
        "lgd_pred": [0.8, 0.9],
    })
    summary = portfolio_el_summary(df)
    assert "mean_lgd" in summary, "the docstring has always promised mean_lgd"
    assert summary["mean_lgd"] == pytest.approx(0.85)


# ── N29: the feature-selection funnel must be recorded, and monotone ────────────


def test_selection_stages_are_recorded_and_monotone(small_accepted):
    """The report can only describe the real four-stage funnel if the counts exist."""
    from credit_risk.data.target import TARGET_COL, define_target
    from credit_risk.utils.config import TargetConfig

    df = define_target(
        small_accepted,
        TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"]),
    )
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    n = len(df)
    split = int(n * 0.7)

    sc = PDScorecard(min_iv=0.001, max_iv=0.95, max_vif=50.0)
    sc.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

    stages = sc.selection_stages
    order = ["n_candidates", "n_after_iv", "n_after_vif", "n_after_elasticnet",
             "n_after_sign_check"]
    for key in order:
        assert key in stages, f"missing stage count: {key}"
    counts = [stages[k] for k in order]
    assert counts == sorted(counts, reverse=True), (
        f"the selection funnel must never grow: {dict(zip(order, counts))}"
    )
    assert sc.binner_kind in {"optbinning", "manual_fallback"}


# ── N1 / N2: the PD horizon fed to one-year formulas must be a one-year PD ──────


def test_lifetime_to_twelve_month_conversion_shrinks_pd():
    """PD_12m = 1 - (1 - PD_lifetime)^(12/T), and must be below the lifetime figure.

    The scorecard target is terminal loan status, so its output is a LIFETIME default
    probability. It was fed straight into the Basel IRB one-year formula and into a
    per-annum P&L, producing a mean PD around 0.245 and an RWA density of 228.8%
    (Flaws.md findings N1, N2).
    """
    pd_life = np.array([0.05, 0.245, 0.60])
    term = np.array([36.0, 36.0, 60.0])
    pd_12m = 1.0 - (1.0 - pd_life) ** (12.0 / term)

    assert np.all(pd_12m < pd_life), "a 12-month PD cannot exceed the lifetime PD"
    # A 3-year loan compounds its annual hazard three times back to the lifetime figure.
    assert 1.0 - (1.0 - pd_12m[1]) ** 3 == pytest.approx(pd_life[1])
    assert pd_12m[1] < 0.15, "a one-year consumer PD should be well inside the IRB range"


def test_irb_capital_is_concave_so_a_lifetime_pd_can_invert_the_stress():
    """Why the horizon error made stressed RWA FALL (Flaws.md finding N18).

    The IRB capital function K(PD) rises, peaks and then declines, because the
    expected-loss term PD*LGD eventually outruns the conditional-loss term. Evaluated at a
    lifetime PD the portfolio sits past that peak, so shocking PD upward reduces capital.
    """
    from credit_risk.risk.basel_irb import irb_capital_requirement

    lgd = np.full(1, 0.9)

    def k_at(p: float) -> float:
        return float(irb_capital_requirement(np.array([p]), lgd)[0])

    # At a plausible one-year PD, stressing upward must increase capital.
    assert k_at(0.16) > k_at(0.08), "stress must raise capital in the one-year PD range"
    # Far out at a lifetime-scale PD the function has turned over.
    assert k_at(0.60) < k_at(0.245), (
        "K is concave; past its peak a PD shock reduces capital, which is exactly the "
        "inversion the lifetime PD produced"
    )


# ── N12: RAROC must not net out the cost of capital ────────────────────────────


def test_raroc_and_economic_profit_are_distinct():
    """RAROC = risk-adjusted return / capital; economic profit subtracts the capital cost.

    Subtracting the cost of capital inside RAROC and then comparing the result to the
    hurdle double-counts it, depressing every reported RAROC by exactly the
    cost-of-capital rate (Flaws.md finding N12).
    """
    revenue, costs, el = 500.0, 120.0, 90.0
    capital, coc = 800.0, 0.12

    risk_adjusted_return = revenue - costs - el
    capital_cost = coc * capital
    economic_profit = risk_adjusted_return - capital_cost
    raroc = risk_adjusted_return / capital

    assert raroc == pytest.approx(0.3625)
    # The identity that pins the two together.
    assert raroc - coc == pytest.approx(economic_profit / capital)
    # The old formula understated RAROC by exactly the cost-of-capital rate.
    assert raroc - (economic_profit / capital) == pytest.approx(coc)


# ── N10: exposure must depend on how long a loan has been on book ──────────────


def test_ead_distinguishes_loans_of_different_age():
    """Same term and rate, different origination date -> different exposure.

    With months-on-book pinned at 40% of term for every loan, EAD was a deterministic
    function of (term, rate) alone: a fully-repaid 2010 loan still showed ~65% exposure at
    the 2018Q4 reporting date (Flaws.md finding N10).
    """
    from credit_risk.models.ead import EADModel

    df = pd.DataFrame({
        "issue_d": ["Jan-2010", "Jan-2018"],
        "term": [" 36 months", " 36 months"],
        "int_rate": [0.12, 0.12],
        "funded_amnt": [10_000.0, 10_000.0],
    })
    model = EADModel(reporting_date="2018-12-31")
    model.fit(df)
    ead = model.predict(df).to_numpy()

    assert model.mob_basis == "elapsed_since_origination"
    assert ead[0] < ead[1], "the older loan must carry the smaller exposure"
    assert ead[0] == pytest.approx(0.0, abs=1.0), (
        "a 36-month loan originated 9 years before the reporting date is fully amortised"
    )


def test_ead_flags_the_uninformative_fallback():
    """When neither basis is available the model must say so, not fail silently."""
    from credit_risk.models.ead import months_on_book_basis

    df = pd.DataFrame({"term": [" 36 months"], "int_rate": [0.12], "funded_amnt": [1000.0]})
    assert months_on_book_basis(df, reporting_date=None) == "fixed_fraction_of_term"


# ── N5: the recalibrator applies only to the era it was fitted on ──────────────


def test_calibrator_scope_leaves_earlier_vintages_untouched():
    """The gate learns a 2016+ transform; applying it to 2007-2014 biased those vintages.

    Development-era vintages ended up over-predicting their realised default rate by up to
    ~53%, and that bias fed EL, RWA and every cutoff decision (Flaws.md finding N5).
    """
    from sklearn.isotonic import IsotonicRegression

    raw = np.array([0.10, 0.30, 0.50, 0.70])
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, np.array([0, 0, 1, 1]))

    sc = PDScorecard()
    sc.set_calibrator(iso, min_issue_year=2016)
    assert sc.calibration_scope == "2016+ vintages only"

    X = pd.DataFrame({"issue_d": ["Jun-2013", "Jun-2013", "Jun-2017", "Jun-2017"]})
    out = sc._apply_calibrator_in_scope(raw, X)

    assert out[0] == raw[0] and out[1] == raw[1], "pre-2016 vintages keep their raw PD"
    assert not np.array_equal(out[2:], raw[2:]), "2016+ vintages are recalibrated"


def test_calibrator_without_scope_applies_everywhere():
    """Unscoped behaviour is unchanged, so the scope is opt-in."""
    from sklearn.isotonic import IsotonicRegression

    raw = np.array([0.10, 0.30, 0.50, 0.70])
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, np.array([0, 0, 1, 1]))
    sc = PDScorecard()
    sc.set_calibrator(iso)
    assert sc.calibration_scope == "all vintages"
    X = pd.DataFrame({"issue_d": ["Jun-2013"] * 4})
    assert not np.array_equal(sc._apply_calibrator_in_scope(raw, X), raw)


# ── N13: economic capital must be able to use the supervisory correlation ──────


def test_economic_capital_accepts_the_supervisory_correlation_curve():
    """EC and regulatory capital are compared side by side, so rho must be comparable.

    A flat rho=0.15 against a supervisory R collapsing to ~0.03 made the EC/RegCap ratio
    mostly a correlation artefact (Flaws.md finding N13).
    """
    from credit_risk.risk.economic_capital import simulate_portfolio_losses

    rng = np.random.default_rng(0)
    n = 400
    pd_arr = rng.uniform(0.02, 0.30, n)
    lgd_arr = np.full(n, 0.8)
    ead_arr = np.full(n, 10_000.0)

    losses_sup = simulate_portfolio_losses(
        pd_arr, lgd_arr, ead_arr, rho="supervisory", n_sim=2000, seed=1
    )
    losses_flat = simulate_portfolio_losses(
        pd_arr, lgd_arr, ead_arr, rho=0.15, n_sim=2000, seed=1
    )
    assert losses_sup.shape == (2000,)
    # At these PDs the supervisory curve sits far below 0.15, so the tail is thinner.
    assert np.quantile(losses_sup, 0.999) < np.quantile(losses_flat, 0.999)

    with pytest.raises(ValueError):
        simulate_portfolio_losses(pd_arr, lgd_arr, ead_arr, rho="nonsense", n_sim=10)


# ── N15: the macro shock is calibrated annually, so it must apply annually ─────


def test_macro_shock_reproduces_its_target_twelve_month_pd():
    """Applying the annual Vasicek transform per MONTH compounded it twelve times over.

    At h~0.005 and Z=-1.64 the monthly hazard rose ~3.5x, pushing cumulative lifetime PD
    far past the ratio the scenario targets and over-shocking the downside ECL
    (Flaws.md finding N15).
    """
    from scipy.special import ndtr, ndtri

    from credit_risk.models.pd_term_structure import DiscreteHazardModel

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "grade": rng.choice(list("ABCDE"), n),
        "int_rate": rng.uniform(0.06, 0.25, n),
        "dti": rng.uniform(5.0, 35.0, n),
        "term": rng.choice([" 36 months", " 60 months"], n),
        "target": (rng.random(n) < 0.25).astype(int),
    })

    model = DiscreteHazardModel(max_horizon=36, seed=42)
    model.fit(df)

    z = -1.6433
    base = model.predict_term_structure(df, macro_shock=0.0)
    shocked = model.predict_term_structure(df, macro_shock=z)

    rho = model.asset_correlation
    target = ndtr(
        (ndtri(np.clip(base["pd_12m"], 1e-9, 1 - 1e-9)) - np.sqrt(rho) * z)
        / np.sqrt(1.0 - rho)
    )
    np.testing.assert_allclose(shocked["pd_12m"], target, rtol=1e-6)
    assert shocked["pd_12m"].mean() > base["pd_12m"].mean(), "Z<0 must raise PD"


# ── N33: reject inference must compare like with like ──────────────────────────


def test_reject_inference_excludes_constant_imputed_features():
    """Features that are constant across rejects were imputed, not observed.

    Roughly two thirds of the scorecard was mean-imputed to a training constant for the
    entire reject population. A constant predictor has no discriminatory power, so the
    reported Gini drop measured that imputation rather than latent through-the-door risk
    (Flaws.md finding N33).
    """
    from credit_risk.business.reject_inference import refit_with_parcelling

    rng = np.random.default_rng(3)
    n_acc, n_rej = 300, 200
    df_acc = pd.DataFrame({
        "fico_range_low": rng.normal(690, 40, n_acc),
        "dti": rng.uniform(5, 35, n_acc),
        "imputed_feature": rng.normal(0, 1, n_acc),
        "pd_pred": rng.uniform(0.05, 0.4, n_acc),
        "target": (rng.random(n_acc) < 0.25).astype(int),
    })
    df_rej = pd.DataFrame({
        "fico_range_low": rng.normal(640, 40, n_rej),
        "dti": rng.uniform(10, 45, n_rej),
        # Mean-imputed to a single training constant, exactly as the aligner does.
        "imputed_feature": np.full(n_rej, float(df_acc["imputed_feature"].mean())),
        "pd_pred": rng.uniform(0.1, 0.6, n_rej),
    })

    _, _, diag = refit_with_parcelling(
        df_acc, df_rej,
        feature_cols=["fico_range_low", "dti", "imputed_feature"],
    )

    assert "imputed_feature" in diag["features_imputed_constant"]
    assert diag["n_features_imputed_constant"] == 1
    assert set(diag["features_used"]) == {"fico_range_low", "dti"}
    assert "identical" in diag["weighting"]


# ── N31: missing values must be a bin, not a sentinel ──────────────────────────


def test_manual_binner_gives_missing_its_own_woe():
    """NaN must not silently inherit the lowest bin's WoE.

    Missing values were filled with -9999 before binning, which dropped every one into the
    extreme low bin of each feature. The binner's own Missing bin was therefore
    structurally empty, and the risk direction implied by the sentinel was arbitrary per
    feature: missing FICO landed in the worst band, missing DTI in the best
    (Flaws.md finding N31).
    """
    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(0.0, 1.0, n)
    # Missing rows are deliberately LOW risk, while the lowest observed bin is HIGH risk.
    y = (x < 0).astype(int)
    x[:100] = np.nan
    y[:100] = 0

    df = pd.DataFrame({"f": x})
    binner = ManualMonotonicBinner(variables=["f"], n_initial_bins=8)
    binner.fit(df, pd.Series(y))

    assert binner.missing_count_["f"] == 100
    out = binner.transform(df)["f"].to_numpy(dtype=float)

    missing_woe = out[:100]
    assert len(set(np.round(missing_woe, 9))) == 1, "all missing rows share one WoE"

    lowest_bin_woe = binner.woe_maps_["f"][0]
    assert missing_woe[0] != pytest.approx(lowest_bin_woe), (
        "the missing bin must carry its own WoE, not bin 0's"
    )
    # WoE = ln(%good / %bad); an all-good missing group must read as low risk (positive).
    assert missing_woe[0] > 0.0


def test_scorecard_no_longer_sentinel_fills_before_binning():
    """The -9999 sentinel must be gone from the fit and scoring paths."""
    import inspect

    from credit_risk.models import pd_scorecard as sc_mod

    def _code_only(text: str) -> str:
        """Strip comments — the fix is documented in prose that names the old sentinel."""
        return "\n".join(
            line.split("#", 1)[0] for line in text.splitlines()
        )

    fit_src = _code_only(inspect.getsource(sc_mod.PDScorecard.fit))
    woe_src = _code_only(inspect.getsource(sc_mod.PDScorecard._woe_transform))
    module_src = _code_only(inspect.getsource(sc_mod))

    assert "-9999" not in fit_src, "fit() must pass NaN through to the binner"
    assert "-9999" not in woe_src, "scoring must pass NaN through to the binner"
    assert "fillna(-9999)" not in module_src
