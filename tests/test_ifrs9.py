"""Tests for IFRS 9 ECL engine and PD term structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.models.pd_term_structure import DiscreteHazardModel
from credit_risk.risk.ifrs9_ecl import (
    IFRS9Config,
    SICRConfig,
    ScenarioConfig,
    assign_stages,
    compute_ecl_single_scenario,
    fit_macro_model,
    run_ifrs9_ecl,
    stage_migration_matrix,
)


# ── Macro model: backward UNRATE sign correction (#4) ─────────────────────────

def _make_macro_dataset(tmp_path):
    """Build a train frame + macro CSV where the RAW default/UNRATE relationship
    is NEGATIVE (higher unemployment -> lower realised default), reproducing the
    charge-off-lag/underwriting-drift confound the sign priors must correct. The
    other macro factors are independent noise (no collinearity) so the negative
    UNRATE sign is a genuine partial effect, not a rank-deficiency artefact."""
    quarters = [f"{y}Q{q}" for y in (2010, 2011, 2012, 2013) for q in (1, 2, 3, 4)]
    rng = np.random.default_rng(7)
    m = len(quarters)
    unrate = rng.uniform(4.0, 10.0, m)                       # independent
    default_rate = np.clip(0.20 - 0.015 * unrate + rng.normal(0, 0.004, m), 0.01, 0.5)
    gdp = rng.uniform(0.0, 4.0, m)                           # independent noise
    fedfunds = rng.uniform(0.1, 2.0, m)                      # independent noise
    cpi = rng.uniform(1.0, 3.0, m)                           # independent noise

    rows = []
    for q, dr in zip(quarters, default_rate):
        year, qtr = int(q[:4]), int(q[-1])
        month = {1: "Feb", 2: "May", 3: "Aug", 4: "Nov"}[qtr]
        n = 400
        target = (rng.random(n) < dr).astype(int)
        rows.extend({"issue_d": f"{month}-{year}", "target": int(t)} for t in target)
    df_train = pd.DataFrame(rows)

    macro = pd.DataFrame({
        "quarter": quarters,
        "UNRATE": unrate,
        "GDP_growth": gdp,
        "FEDFUNDS": fedfunds,
        "CPI_inflation": cpi,
    })
    macro_path = tmp_path / "macro_quarterly.csv"
    macro.to_csv(macro_path, index=False)
    return df_train, str(macro_path)


class TestMacroSignCorrection:
    def test_raw_unrate_sign_is_negative(self, tmp_path) -> None:
        """Sanity: the raw contemporaneous OLS really does produce a negative sign."""
        df_train, macro_path = _make_macro_dataset(tmp_path)
        shocks = fit_macro_model(df_train, macro_path, unrate_lag=0, enforce_sign_priors=True)
        assert shocks["elasticities"]["UNRATE"] < 0.0

    def test_adjusted_unrate_sign_is_positive(self, tmp_path) -> None:
        df_train, macro_path = _make_macro_dataset(tmp_path)
        shocks = fit_macro_model(df_train, macro_path, unrate_lag=0, enforce_sign_priors=True)
        assert shocks["macro_sign_adjusted"] is True
        assert shocks["elasticities_adjusted"]["UNRATE"] >= 0.0
        assert shocks["elasticities_adjusted"]["GDP_growth"] <= 0.0

    def test_scenario_ordering_downside_highest(self, tmp_path) -> None:
        df_train, macro_path = _make_macro_dataset(tmp_path)
        shocks = fit_macro_model(df_train, macro_path, unrate_lag=0, enforce_sign_priors=True)
        preds = shocks["predictions"]
        assert preds["downside"] >= preds["baseline"] >= preds["upside"]
        # Vasicek convention: adverse (downside) -> Z < 0
        assert shocks["downside"] <= shocks["baseline"] <= shocks["upside"]


# ── Macro model: optional 5th variable (HPI_growth) ───────────────────────────

def _make_macro_dataset_with_hpi(tmp_path):
    """Same as _make_macro_dataset but with a 5th column, HPI_growth, so we can
    verify fit_macro_model picks it up automatically when present (and applies
    its sign prior) without breaking the 4-column path used elsewhere."""
    quarters = [f"{y}Q{q}" for y in (2010, 2011, 2012, 2013) for q in (1, 2, 3, 4)]
    rng = np.random.default_rng(7)
    m = len(quarters)
    unrate = rng.uniform(4.0, 10.0, m)
    default_rate = np.clip(0.20 - 0.015 * unrate + rng.normal(0, 0.004, m), 0.01, 0.5)
    gdp = rng.uniform(0.0, 4.0, m)
    fedfunds = rng.uniform(0.1, 2.0, m)
    cpi = rng.uniform(1.0, 3.0, m)
    hpi = rng.uniform(-2.0, 4.0, m)  # independent noise, like the other factors

    rows = []
    for q, dr in zip(quarters, default_rate):
        year, qtr = int(q[:4]), int(q[-1])
        month = {1: "Feb", 2: "May", 3: "Aug", 4: "Nov"}[qtr]
        n = 400
        target = (rng.random(n) < dr).astype(int)
        rows.extend({"issue_d": f"{month}-{year}", "target": int(t)} for t in target)
    df_train = pd.DataFrame(rows)

    macro = pd.DataFrame({
        "quarter": quarters,
        "UNRATE": unrate,
        "GDP_growth": gdp,
        "FEDFUNDS": fedfunds,
        "CPI_inflation": cpi,
        "HPI_growth": hpi,
    })
    macro_path = tmp_path / "macro_quarterly_hpi.csv"
    macro.to_csv(macro_path, index=False)
    return df_train, str(macro_path)


class TestMacroHPIVariable:
    def test_hpi_growth_used_when_present(self, tmp_path) -> None:
        df_train, macro_path = _make_macro_dataset_with_hpi(tmp_path)
        shocks = fit_macro_model(df_train, macro_path, unrate_lag=0, enforce_sign_priors=True)
        assert "HPI_growth" in shocks["elasticities"]
        assert "HPI_growth" in shocks["elasticities_adjusted"]
        # Sign prior: rising home prices -> lower default rate.
        assert shocks["elasticities_adjusted"]["HPI_growth"] <= 0.0

    def test_scenario_ordering_holds_with_hpi(self, tmp_path) -> None:
        df_train, macro_path = _make_macro_dataset_with_hpi(tmp_path)
        shocks = fit_macro_model(df_train, macro_path, unrate_lag=0, enforce_sign_priors=True)
        preds = shocks["predictions"]
        assert preds["downside"] >= preds["baseline"] >= preds["upside"]
        assert shocks["downside"] <= shocks["baseline"] <= shocks["upside"]

    def test_four_column_csv_still_works(self, tmp_path) -> None:
        """The pre-HPI 4-column macro CSV path must remain unaffected."""
        df_train, macro_path = _make_macro_dataset(tmp_path)
        shocks = fit_macro_model(df_train, macro_path, unrate_lag=0, enforce_sign_priors=True)
        assert "HPI_growth" not in shocks["elasticities"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_portfolio() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "grade": rng.choice(["A", "B", "C", "D", "E"], n),
        "int_rate": rng.uniform(5.0, 25.0, n),
        "dti": rng.uniform(5.0, 40.0, n),
        "term": rng.choice(["36 months", "60 months"], n),
        "target": rng.binomial(1, 0.15, n),
        "funded_amnt": rng.uniform(5000, 35000, n),
    })


@pytest.fixture()
def fitted_hazard(small_portfolio: pd.DataFrame) -> DiscreteHazardModel:
    model = DiscreteHazardModel(max_horizon=36, seed=42)
    model.fit(small_portfolio)
    return model


# ── Term structure tests ───────────────────────────────────────────────────────

class TestDiscreteHazardModel:
    def test_marginal_pd_sums_leq_1(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        ts = fitted_hazard.predict_term_structure(small_portfolio)
        row_sums = ts["marginal_pd"].sum(axis=1)
        assert (row_sums <= 1.0 + 1e-9).all(), "Sum of marginal PDs must be ≤ 1"

    def test_lifetime_pd_geq_12m_pd(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        ts = fitted_hazard.predict_term_structure(small_portfolio)
        assert (ts["pd_lifetime"] >= ts["pd_12m"] - 1e-9).all(), (
            "Lifetime PD must be ≥ 12m PD"
        )

    def test_pd_in_0_1(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        ts = fitted_hazard.predict_term_structure(small_portfolio)
        assert (ts["pd_12m"] >= 0.0).all() and (ts["pd_12m"] <= 1.0).all()
        assert (ts["pd_lifetime"] >= 0.0).all() and (ts["pd_lifetime"] <= 1.0).all()

    def test_survival_monotone_decreasing(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        ts = fitted_hazard.predict_term_structure(small_portfolio)
        diffs = np.diff(ts["survival"], axis=1)
        assert (diffs <= 1e-9).all(), "Survival must be non-increasing over time"

    def test_adverse_z_increases_pd(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        # Eq. 15 convention: Z < 0 = recession = higher PD
        ts_base = fitted_hazard.predict_term_structure(small_portfolio, macro_shock=0.0)
        ts_stress = fitted_hazard.predict_term_structure(small_portfolio, macro_shock=-2.0)
        assert ts_stress["pd_lifetime"].mean() > ts_base["pd_lifetime"].mean()

    def test_favourable_z_decreases_pd(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        # Eq. 15 convention: Z > 0 = expansion = lower PD
        ts_base = fitted_hazard.predict_term_structure(small_portfolio, macro_shock=0.0)
        ts_upside = fitted_hazard.predict_term_structure(small_portfolio, macro_shock=2.0)
        assert ts_upside["pd_lifetime"].mean() < ts_base["pd_lifetime"].mean()

    def test_z_ordering_full_convention(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        # PD(Z=-2) > PD(0) > PD(Z=+2) — Fix 1.2 acceptance check
        pd_rec = fitted_hazard.predict_term_structure(small_portfolio, macro_shock=-2.0)["pd_lifetime"].mean()
        pd_base = fitted_hazard.predict_term_structure(small_portfolio, macro_shock=0.0)["pd_lifetime"].mean()
        pd_exp = fitted_hazard.predict_term_structure(small_portfolio, macro_shock=2.0)["pd_lifetime"].mean()
        assert pd_rec > pd_base > pd_exp


# ── Staging tests ──────────────────────────────────────────────────────────────

class TestAssignStages:
    def test_defaults_get_stage_3(self) -> None:
        df = pd.DataFrame({"target": [1, 0, 0, 1]})
        pd_curr = np.array([0.5, 0.1, 0.05, 0.8])
        pd_orig = np.array([0.1, 0.1, 0.05, 0.1])
        stages = assign_stages(df, pd_curr, pd_orig, SICRConfig())
        assert stages[0] == 3
        assert stages[3] == 3

    def test_sicr_threshold_gives_stage_2(self) -> None:
        df = pd.DataFrame({"target": [0, 0, 0]})
        pd_orig = np.array([0.05, 0.05, 0.05])
        # First loan exceeds 2.5× multiplier; second hits absolute threshold
        pd_curr = np.array([0.15, 0.05, 0.05])
        stages = assign_stages(df, pd_curr, pd_orig, SICRConfig(pd_multiplier=2.5))
        assert stages[0] == 2  # 0.15 > 0.05 × 2.5 = 0.125
        assert stages[2] == 1  # no change

    def test_performing_loans_are_stage_1(self) -> None:
        df = pd.DataFrame({"target": [0] * 5})
        pd_curr = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
        pd_orig = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
        stages = assign_stages(df, pd_curr, pd_orig, SICRConfig(pd_multiplier=2.5))
        assert (stages == 1).all()

    def test_absolute_threshold_gives_stage_2(self) -> None:
        df = pd.DataFrame({"target": [0]})
        pd_curr = np.array([0.25])  # > abs_threshold=0.20
        pd_orig = np.array([0.22])  # only 1.14× — below multiplier
        stages = assign_stages(df, pd_curr, pd_orig, SICRConfig(pd_multiplier=2.5, abs_threshold=0.20))
        assert stages[0] == 2


# ── ECL formula tests ─────────────────────────────────────────────────────────

class TestECLFormula:
    def test_stage3_ecl_equals_lgd_times_ead(self) -> None:
        """Stage 3: PD = 1, ECL = LGD × EAD (no discounting)."""
        n, T = 5, 24
        marginal_pd = np.zeros((n, T))
        marginal_pd[:, 0] = 1.0  # immediate default
        lgd = np.full(n, 0.45)
        ead = np.full(n, 10000.0)
        eir = np.full(n, 0.01)
        stages = np.full(n, 3)

        ecl = compute_ecl_single_scenario(marginal_pd, lgd, ead, eir, stages)
        expected = lgd * ead
        np.testing.assert_allclose(ecl, expected, rtol=1e-10)

    def test_ecl_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        n, T = 50, 36
        marginal_pd = rng.random((n, T)) * 0.05
        lgd = rng.random(n) * 0.7
        ead = rng.random(n) * 50000
        eir = rng.random(n) * 0.02
        stages = rng.integers(1, 4, n)
        ecl = compute_ecl_single_scenario(marginal_pd, lgd, ead, eir, stages)
        assert (ecl >= 0.0).all()

    def test_stage1_ecl_leq_stage2_ecl(self) -> None:
        """Same loan: Stage 2 (lifetime) ECL ≥ Stage 1 (12m) ECL."""
        n, T = 10, 36
        rng = np.random.default_rng(7)
        marginal_pd = rng.random((n, T)) * 0.05
        lgd = np.full(n, 0.45)
        ead = np.full(n, 10000.0)
        eir = np.full(n, 0.01)

        ecl_s1 = compute_ecl_single_scenario(marginal_pd, lgd, ead, eir, np.ones(n, int))
        ecl_s2 = compute_ecl_single_scenario(marginal_pd, lgd, ead, eir, np.full(n, 2))
        assert (ecl_s2 >= ecl_s1 - 1e-9).all()


# ── Full pipeline tests ────────────────────────────────────────────────────────

class TestRunIFRS9ECL:
    def test_ecl_columns_present(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        lgd = np.full(len(small_portfolio), 0.45)
        ead = small_portfolio["funded_amnt"].values
        result = run_ifrs9_ecl(small_portfolio, fitted_hazard, lgd, ead)
        for col in ["stage", "pd_12m", "pd_lifetime", "ecl", "ecl_s1", "ecl_s2", "ecl_s3"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_stage_values_are_1_2_3(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        lgd = np.full(len(small_portfolio), 0.45)
        ead = small_portfolio["funded_amnt"].values
        result = run_ifrs9_ecl(small_portfolio, fitted_hazard, lgd, ead)
        assert set(result["stage"].unique()).issubset({1, 2, 3})

    def test_ecl_nonnegative(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        lgd = np.full(len(small_portfolio), 0.45)
        ead = small_portfolio["funded_amnt"].values
        result = run_ifrs9_ecl(small_portfolio, fitted_hazard, lgd, ead)
        assert (result["ecl"] >= 0.0).all()

    def test_ecl_stage_sum_equals_total(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        lgd = np.full(len(small_portfolio), 0.45)
        ead = small_portfolio["funded_amnt"].values
        result = run_ifrs9_ecl(small_portfolio, fitted_hazard, lgd, ead)
        total = float(result["ecl"].sum())
        stage_sum = float(result["ecl_s1"].sum() + result["ecl_s2"].sum() + result["ecl_s3"].sum())
        assert abs(total - stage_sum) < 1.0, "Stage ECL sums must equal total ECL"

    def test_baseline_only_scenario_equals_ecl(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        """Weighted ECL with only baseline scenario (w=1.0) matches baseline ECL column."""
        cfg = IFRS9Config(
            scenarios=[ScenarioConfig("baseline", 1.0, 0.0)],
            sicr=SICRConfig(),
        )
        lgd = np.full(len(small_portfolio), 0.45)
        ead = small_portfolio["funded_amnt"].values
        result = run_ifrs9_ecl(small_portfolio, fitted_hazard, lgd, ead, cfg=cfg)
        np.testing.assert_allclose(
            result["ecl"].values, result["ecl_baseline"].values, rtol=1e-9
        )

    def test_summary_attrs_set(
        self, fitted_hazard: DiscreteHazardModel, small_portfolio: pd.DataFrame
    ) -> None:
        lgd = np.full(len(small_portfolio), 0.45)
        ead = small_portfolio["funded_amnt"].values
        result = run_ifrs9_ecl(small_portfolio, fitted_hazard, lgd, ead)
        assert "ifrs9_summary" in result.attrs
        summary = result.attrs["ifrs9_summary"]
        assert "total_ecl" in summary
        assert "coverage_ratio" in summary


# ── Stage migration matrix ─────────────────────────────────────────────────────

class TestStageMigrationMatrix:
    def test_diagonal_only(self) -> None:
        stages = np.array([1, 2, 3, 1, 2])
        mat = stage_migration_matrix(stages, stages)
        assert mat.loc[1, 1] == 2
        assert mat.loc[2, 2] == 2
        assert mat.loc[3, 3] == 1
        assert mat.loc[1, 2] == 0

    def test_shape_3x3(self) -> None:
        s0 = np.array([1, 1, 2, 3])
        s1 = np.array([1, 2, 3, 3])
        mat = stage_migration_matrix(s0, s1)
        assert mat.shape == (3, 3)


def test_marginal_pd_sums_to_lifetime_pd():
    import numpy as np
    h = np.full(36, 0.02)
    S = np.cumprod(1 - h)
    S_lag = np.concatenate([[1.0], S[:-1]])
    m = S_lag * h
    assert abs(m.sum() - (1 - S[-1])) < 1e-10


def test_survival_monotone_decreasing():
    import numpy as np
    h = np.array([0.01, 0.015, 0.02, 0.025] * 9)  # increasing hazard
    S = np.cumprod(1 - h)
    assert np.all(np.diff(S) <= 0), "Survival function must be monotone decreasing"

