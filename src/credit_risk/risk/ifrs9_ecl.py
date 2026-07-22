"""IFRS 9 Expected Credit Loss (ECL) engine.

Three-stage classification:
    Stage 1 (performing):     12-month ECL
    Stage 2 (SICR):           lifetime ECL (significant increase in credit risk)
    Stage 3 (default/credit-impaired): lifetime ECL, PD = 1.0

SICR triggers:
    - Relative PD threshold: lifetime_pd > origination_lifetime_pd × sicr_pd_mult
    - Absolute PD floor: lifetime_pd > sicr_abs_threshold
    - 30+ DPD backstop (if delinquency info available)

ECL formula (Appendix C):
    ECL = Σ_t  MarginalPD(t) · LGD(t) · EAD(t) · DF(t)

    DF(t) = 1 / (1 + EIR)^t      EIR per loan from int_rate / 12

Macro scenarios (weighted average):
    ECL_final = Σ_s w_s · ECL_s

    Scenarios: baseline, upside, downside
    Each scenario applies a different macro_shock to the hazard model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_int_rate_to_fraction(int_rate: np.ndarray) -> np.ndarray:
    """Convert interest rates to a fraction, per-value.

    Consumer loan interest rates are always well below 100% whether stored
    as a percentage (e.g. 12.5) or a fraction (e.g. 0.125); values above 1.0
    are therefore unambiguously percent-scale. A per-value check (rather
    than a population `.mean() > 1.0` check) avoids misclassifying an
    entire column when rates mix scales or skew low.
    """
    int_rate = np.asarray(int_rate, dtype=float)
    return np.where(int_rate > 1.0, int_rate / 100.0, int_rate)


@dataclass
class ScenarioConfig:
    name: str
    weight: float
    macro_shock: float


@dataclass
class SICRConfig:
    pd_multiplier: float = 2.5
    abs_threshold: float = 0.20
    dpd_backstop: int = 30


@dataclass
class IFRS9Config:
    # macro_shock is the Vasicek systematic factor Z (Eq. 15 convention):
    # Z < 0 = adverse shock (recession, higher PD); Z > 0 = favourable shock.
    scenarios: list[ScenarioConfig] = field(default_factory=lambda: [
        ScenarioConfig("baseline", 0.50, 0.0),
        ScenarioConfig("upside", 0.25, 0.5),
        ScenarioConfig("downside", 0.25, -1.0),
    ])
    sicr: SICRConfig = field(default_factory=SICRConfig)


def _discount_factors(eir_monthly: np.ndarray, horizon: int) -> np.ndarray:
    """Compute discount factor matrix DF[i, t] = 1/(1+EIR_i)^t.

    Parameters
    ----------
    eir_monthly:
        Monthly EIR per loan, shape (n,).
    horizon:
        Number of time steps.

    Returns
    -------
    np.ndarray shape (n, horizon)
    """
    t_vec = np.arange(1, horizon + 1, dtype=float)
    return 1.0 / (1.0 + eir_monthly[:, None]) ** t_vec


def assign_stages(
    df: pd.DataFrame,
    pd_lifetime: np.ndarray,
    pd_orig_lifetime: np.ndarray | None,
    sicr_cfg: SICRConfig,
    in_default: np.ndarray | None = None,
) -> np.ndarray:
    """Assign IFRS 9 stages (1, 2, 3) per loan.

    Parameters
    ----------
    df:
        Portfolio DataFrame.
    pd_lifetime:
        Current lifetime PD per loan.
    pd_orig_lifetime:
        Lifetime PD at origination. If None, all go to Stage 1/3 only.
    sicr_cfg:
        SICR configuration.
    in_default:
        Boolean array indicating current default. If None, uses target col.

    Returns
    -------
    np.ndarray of int (1, 2, or 3), shape (n,)
    """
    n = len(df)
    stages = np.ones(n, dtype=int)  # default Stage 1

    # Stage 3: currently in default
    if in_default is not None:
        default_mask = np.asarray(in_default, dtype=bool)
    elif "target" in df.columns:
        default_mask = df["target"].fillna(0).astype(bool).values
    else:
        default_mask = np.zeros(n, dtype=bool)

    stages[default_mask] = 3

    # Stage 2: SICR (only for non-default)
    non_default = ~default_mask
    if pd_orig_lifetime is not None and non_default.any():
        pd_orig = np.asarray(pd_orig_lifetime, dtype=float)
        pd_curr = np.asarray(pd_lifetime, dtype=float)
        relative_sicr = pd_curr > pd_orig * sicr_cfg.pd_multiplier
        absolute_sicr = pd_curr > sicr_cfg.abs_threshold
        sicr_mask = non_default & (relative_sicr | absolute_sicr)

        # 30+ DPD backstop
        if "delinq_2yrs" in df.columns:
            dpd_mask = non_default & (
                pd.to_numeric(df["delinq_2yrs"], errors="coerce").fillna(0) >= 1
            )
            sicr_mask = sicr_mask | dpd_mask

        stages[sicr_mask] = 2

    return stages


def compute_ecl_single_scenario(
    marginal_pd: np.ndarray,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    eir_monthly: np.ndarray,
    stages: np.ndarray,
) -> np.ndarray:
    """Compute ECL per loan for a single macro scenario.

    Parameters
    ----------
    marginal_pd:
        Shape (n, horizon) marginal PD from term structure.
    lgd_arr:
        LGD per loan, shape (n,).
    ead_arr:
        EAD per loan, shape (n,).
    eir_monthly:
        Monthly discount rate per loan.
    stages:
        IFRS 9 stage per loan.

    Returns
    -------
    np.ndarray shape (n,) ECL per loan.
    """
    n, horizon = marginal_pd.shape
    df_mat = _discount_factors(eir_monthly, horizon)

    # ECL per time step: MarginalPD(t) × LGD × EAD × DF(t)
    lgd_col = lgd_arr[:, None]  # (n, 1)
    ead_col = ead_arr[:, None]

    ecl_steps = marginal_pd * lgd_col * ead_col * df_mat  # (n, horizon)

    # Stage 1: 12-month ECL (sum first 12 steps)
    # Stage 2/3: lifetime ECL (sum all steps)
    ecl_12m = ecl_steps[:, :12].sum(axis=1) if horizon >= 12 else ecl_steps.sum(axis=1)
    ecl_lifetime = ecl_steps.sum(axis=1)

    # Stage 3 override: PD = 1 → ECL = LGD × EAD (no discounting on certain loss)
    ecl_stage3 = lgd_arr * ead_arr

    ecl = np.where(stages == 1, ecl_12m,
           np.where(stages == 3, ecl_stage3, ecl_lifetime))
    return np.maximum(ecl, 0.0)


def run_ifrs9_ecl(
    df: pd.DataFrame,
    hazard_model: object,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    cfg: IFRS9Config | None = None,
    pd_orig_lifetime: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full IFRS 9 ECL computation with macro scenario weighting.

    Parameters
    ----------
    df:
        Portfolio DataFrame.
    hazard_model:
        Fitted `DiscreteHazardModel` instance.
    lgd_arr:
        LGD per loan.
    ead_arr:
        EAD per loan.
    cfg:
        IFRS9 configuration. Uses defaults if None.
    pd_orig_lifetime:
        Lifetime PD at origination for SICR. If None, only Stage 3 assigned.

    Returns
    -------
    pd.DataFrame
        Input df with added columns: stage, pd_12m, pd_lifetime, ecl,
        ecl_stage1, ecl_stage2, ecl_stage3, plus per-scenario ECL.
    """
    if cfg is None:
        cfg = IFRS9Config()

    out = df.copy()
    lgd = np.asarray(lgd_arr, dtype=float)
    ead = np.asarray(ead_arr, dtype=float)

    # Monthly EIR from annual int_rate
    int_rate = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
    # int_rate may be stored as percent (e.g., 12.5) or fraction (0.125)
    int_rate = normalize_int_rate_to_fraction(int_rate)
    eir_monthly = int_rate / 12.0

    # Validate scenario weights sum to 1
    total_w = sum(s.weight for s in cfg.scenarios)
    if abs(total_w - 1.0) > 1e-6:
        logger.warning("Scenario weights sum to %.4f, not 1.0. Normalising.", total_w)
        for s in cfg.scenarios:
            s.weight /= total_w

    # Baseline term structure for staging
    baseline_ts = hazard_model.predict_term_structure(df, macro_shock=0.0)
    pd_lifetime = baseline_ts["pd_lifetime"]
    pd_12m = baseline_ts["pd_12m"]

    stages = assign_stages(df, pd_lifetime, pd_orig_lifetime, cfg.sicr)
    out["stage"] = stages
    out["pd_12m"] = pd_12m
    out["pd_lifetime"] = pd_lifetime

    # Weighted ECL across scenarios
    ecl_weighted = np.zeros(len(df))
    for scenario in cfg.scenarios:
        ts = hazard_model.predict_term_structure(df, macro_shock=scenario.macro_shock)
        ecl_s = compute_ecl_single_scenario(
            ts["marginal_pd"], lgd, ead, eir_monthly, stages
        )
        out[f"ecl_{scenario.name}"] = ecl_s
        ecl_weighted += scenario.weight * ecl_s

    out["ecl"] = ecl_weighted

    # Stage-level ECL columns
    out["ecl_s1"] = np.where(stages == 1, ecl_weighted, 0.0)
    out["ecl_s2"] = np.where(stages == 2, ecl_weighted, 0.0)
    out["ecl_s3"] = np.where(stages == 3, ecl_weighted, 0.0)

    # Coverage ratio (ECL / EAD)
    out["ecl_coverage"] = ecl_weighted / np.maximum(ead, 1.0)

    # Portfolio summary
    total_ecl = float(ecl_weighted.sum())
    total_ead = float(ead.sum())
    coverage = total_ecl / total_ead if total_ead > 0 else 0.0
    stage_counts = pd.Series(stages).value_counts().to_dict()

    logger.info(
        "IFRS 9 ECL: total=%.2f | coverage=%.4f%% | stages=%s",
        total_ecl, coverage * 100, stage_counts,
    )

    out.attrs["ifrs9_summary"] = {
        "total_ecl": total_ecl,
        "total_ead": total_ead,
        "coverage_ratio": coverage,
        "stage_counts": stage_counts,
        "ecl_by_stage": {
            "s1": float(out["ecl_s1"].sum()),
            "s2": float(out["ecl_s2"].sum()),
            "s3": float(out["ecl_s3"].sum()),
        },
    }

    return out


def stage_migration_matrix(
    stages_t0: np.ndarray,
    stages_t1: np.ndarray,
) -> pd.DataFrame:
    """Compute stage migration matrix from origination to reporting date.

    Returns
    -------
    pd.DataFrame (3×3) with row = from-stage, col = to-stage, values = counts.
    """
    idx = [1, 2, 3]
    mat = pd.DataFrame(0, index=idx, columns=idx)
    for s0, s1 in zip(stages_t0, stages_t1):
        s0, s1 = int(s0), int(s1)
        if s0 in idx and s1 in idx:
            mat.loc[s0, s1] += 1
    mat.index.name = "from_stage"
    mat.columns.name = "to_stage"
    return mat


def ecl_scenario_sensitivity(
    df: pd.DataFrame,
    hazard_model: object,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    macro_shocks: list[float],
    stages: np.ndarray,
) -> pd.DataFrame:
    """ECL sensitivity to macro shock values.

    Returns
    -------
    pd.DataFrame with columns: macro_shock, total_ecl, coverage_ratio.
    """
    int_rate = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
    int_rate = normalize_int_rate_to_fraction(int_rate)
    eir_monthly = int_rate / 12.0
    ead = np.asarray(ead_arr, dtype=float)
    lgd = np.asarray(lgd_arr, dtype=float)
    total_ead = ead.sum()

    rows = []
    for shock in macro_shocks:
        ts = hazard_model.predict_term_structure(df, macro_shock=shock)
        ecl = compute_ecl_single_scenario(
            ts["marginal_pd"], lgd, ead, eir_monthly, stages
        )
        total_ecl = float(ecl.sum())
        rows.append({
            "macro_shock": shock,
            "total_ecl": total_ecl,
            "coverage_ratio": total_ecl / total_ead if total_ead > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# Standard PD/LGD/EAD what-if stress scenarios for the ECL sensitivity calculator.
DEFAULT_SHOCK_SCENARIOS: dict[str, dict[str, float]] = {
    "PD +20%": {"pd_multiplier": 1.20},
    "PD +50%": {"pd_multiplier": 1.50},
    "LGD +10pp": {"lgd_add": 0.10},
    "EAD +15%": {"ead_multiplier": 1.15},
    "Combined": {"pd_multiplier": 1.30, "lgd_add": 0.05, "ead_multiplier": 1.10},
    "COVID-like (PD x2.5)": {"pd_multiplier": 2.50, "lgd_add": 0.15},
    "GFC-like (PD x3.0)": {"pd_multiplier": 3.00, "lgd_add": 0.20},
}


def _cap_cumulative_pd(marginal_pd: np.ndarray) -> np.ndarray:
    """Cap each loan's cumulative (lifetime) PD at 1.0 across periods.

    Per-period clipping to [0, 1] does not bound the row sum: a large
    ``pd_multiplier`` stress can scale every period up and still leave each
    one individually <= 1, while the cumulative default probability for the
    loan exceeds 1.0 (a loan cannot default more than once), which in turn
    inflates shocked ECL above the loan's exposure. This trims the period
    where the running total first crosses 1.0 and zeroes any periods after
    it, leaving earlier periods untouched.
    """
    cum = np.cumsum(marginal_pd, axis=1)
    excess = np.clip(cum - 1.0, 0.0, None)
    if not np.any(excess > 0):
        return marginal_pd
    prev_excess = np.hstack([np.zeros((excess.shape[0], 1)), excess[:, :-1]])
    remove = np.clip(excess - prev_excess, 0.0, None)
    return np.clip(marginal_pd - remove, 0.0, None)


def ecl_shock_sensitivity(
    df: pd.DataFrame,
    hazard_model: object,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    stages: np.ndarray,
    shock_scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """What-if ECL calculator under PD / LGD / EAD stress scenarios.

    Each scenario dict may contain ``pd_multiplier`` (scales marginal PD),
    ``lgd_add`` (additive LGD shock in absolute terms) and ``ead_multiplier``
    (scales EAD). The baseline term structure is computed once and re-used, so
    only the stressed inputs change per scenario --- mirroring
    :func:`ecl_scenario_sensitivity`.

    Returns
    -------
    pd.DataFrame with columns: ``scenario, base_ecl, shocked_ecl, delta_ecl, delta_pct``.
    """
    if shock_scenarios is None:
        shock_scenarios = DEFAULT_SHOCK_SCENARIOS

    int_rate = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
    int_rate = normalize_int_rate_to_fraction(int_rate)
    eir_monthly = int_rate / 12.0
    lgd = np.asarray(lgd_arr, dtype=float)
    ead = np.asarray(ead_arr, dtype=float)

    base_ts = hazard_model.predict_term_structure(df, macro_shock=0.0)
    base_marginal = base_ts["marginal_pd"]
    base_ecl = float(
        compute_ecl_single_scenario(base_marginal, lgd, ead, eir_monthly, stages).sum()
    )

    rows = []
    for name, shocks in shock_scenarios.items():
        mp = np.clip(base_marginal * float(shocks.get("pd_multiplier", 1.0)), 0.0, 1.0)
        mp = _cap_cumulative_pd(mp)
        lgd_s = np.clip(lgd + float(shocks.get("lgd_add", 0.0)), 0.0, 1.0)
        ead_s = ead * float(shocks.get("ead_multiplier", 1.0))
        shocked = float(
            compute_ecl_single_scenario(mp, lgd_s, ead_s, eir_monthly, stages).sum()
        )
        rows.append({
            "scenario": name,
            "base_ecl": base_ecl,
            "shocked_ecl": shocked,
            "delta_ecl": shocked - base_ecl,
            "delta_pct": (shocked / base_ecl - 1.0) * 100.0 if base_ecl > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# Economic sign priors for the macro -> default-rate relationship: rising
# unemployment / policy rates / inflation and falling growth all raise defaults.
# HPI_growth (house price growth) follows GDP_growth's direction: rising home
# prices support collateral values and household wealth, lowering defaults.
_MACRO_SIGN_PRIORS = {
    "UNRATE": +1.0, "GDP_growth": -1.0, "FEDFUNDS": +1.0, "CPI_inflation": +1.0,
    "HPI_growth": -1.0,
}

# All macro columns fit_macro_model knows how to use, in preference order.
# Only the ones actually present in the supplied macro CSV are used, so this
# stays backward-compatible with 4-column (pre-HPI) macro_quarterly.csv files.
_ALL_MACRO_COLS = ["UNRATE", "GDP_growth", "FEDFUNDS", "CPI_inflation", "HPI_growth"]

# Per-scenario macro deltas applied on top of the sample-mean baseline.
# (UNRATE/GDP_growth/FEDFUNDS deltas below mirror the pre-existing inline
# logic; HPI_growth deltas are new.)
_SCENARIO_DELTAS = {
    "upside": {"UNRATE": -1.0, "GDP_growth": +1.0, "FEDFUNDS": -0.5, "HPI_growth": +2.0},
    "downside": {"UNRATE": +3.0, "GDP_growth": -3.0, "FEDFUNDS": -1.5, "HPI_growth": -8.0},
}


def fit_macro_model(
    df_train: pd.DataFrame,
    macro_path: str,
    unrate_lag: int = 2,
    enforce_sign_priors: bool = True,
) -> dict:
    """Train OLS regression linking historical quarterly default rates to macro variables.

    The raw contemporaneous regression on this portfolio produces a spurious
    *negative* UNRATE coefficient: charge-offs are recognised with a lag and
    origination underwriting drifts over 2007--2018, so high-unemployment vintages
    (tightly underwritten, 2009-11) show lower realised defaults than the loosely
    underwritten low-unemployment 2015-16 vintages. Two corrections are applied:

    1. ``unrate_lag`` lags the macro series relative to the origination cohort so
       that a cohort's realised default rate is aligned with the macro environment
       that prevails during the loan's life.
    2. ``enforce_sign_priors`` imposes economically-correct signs (magnitude from
       the fitted OLS) on the coefficients used for *scenario projection*, so the
       implied default-rate ordering is guaranteed Downside > Baseline > Upside.

    The raw OLS coefficients and R^2 are still reported honestly; the adjusted
    coefficients actually used for projection are returned separately with a flag.

    Returns a dict with the mapped Vasicek shocks Z per scenario plus diagnostics.
    """
    import statsmodels.api as sm  # noqa: PLC0415
    from scipy.special import ndtri  # noqa: PLC0415

    macro_df = pd.read_csv(macro_path)
    # Only use macro columns actually present in the CSV, so this stays
    # backward-compatible with older 4-column macro_quarterly.csv files
    # that predate HPI_growth.
    macro_cols = [c for c in _ALL_MACRO_COLS if c in macro_df.columns]

    # Calculate default rate per issue quarter in df_train
    df_train = df_train.copy()
    df_train["quarter"] = pd.to_datetime(df_train["issue_d"], format="%b-%Y", errors="coerce").dt.to_period("Q").astype(str)

    # Group by quarter to get default rates
    q_stats = df_train.groupby("quarter")["target"].agg(["count", "sum"])
    q_stats["default_rate"] = q_stats["sum"] / q_stats["count"]
    q_stats = q_stats.reset_index()

    # Lag the macro series relative to the origination cohort (charge-off lag).
    merged = pd.DataFrame()
    if unrate_lag and unrate_lag > 0:
        macro_lag_df = macro_df.sort_values("quarter").reset_index(drop=True).copy()
        present = [c for c in macro_cols if c in macro_lag_df.columns]
        macro_lag_df[present] = macro_lag_df[present].shift(-unrate_lag)
        macro_lag_df = macro_lag_df.dropna(subset=present)
        merged = q_stats.merge(macro_lag_df, on="quarter", how="inner")

    lag_used = unrate_lag if len(merged) >= 4 else 0
    if len(merged) < 4:
        # Fall back to contemporaneous macro if lagging leaves too few points.
        merged = q_stats.merge(macro_df, on="quarter", how="inner")
    if len(merged) < 4:
        logger.warning("Not enough quarterly historical data to fit macro regression. Using fallback shocks.")
        # Z convention: Z < 0 = adverse (recession), Z > 0 = favourable
        return {
            "baseline": 0.0, "upside": 0.5, "downside": -1.0,
            "elasticities": {}, "elasticities_adjusted": {},
            "macro_sign_adjusted": False, "macro_unrate_lag": 0, "r_squared": float("nan"),
        }

    X = merged[macro_cols]
    y = merged["default_rate"]

    # Fit OLS (reported honestly for R^2 and raw coefficients)
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    elasticities = model.params.to_dict()
    logger.info("Macro OLS Regression fitted. R2=%.4f (unrate_lag=%d)", model.rsquared, lag_used)

    # Coefficients used for scenario projection: impose economic sign priors.
    proj_params = dict(elasticities)
    sign_adjusted = False
    if enforce_sign_priors:
        for k, prior in _MACRO_SIGN_PRIORS.items():
            if k in proj_params:
                signed = prior * abs(proj_params[k])
                if signed != proj_params[k]:
                    sign_adjusted = True
                proj_params[k] = signed

    # Define quarterly scenarios around the CENTRAL (expected) macro environment.
    # Using the sample-mean macro state — not the last training quarter — anchors the
    # baseline to through-the-cycle conditions, so the baseline systematic factor sits
    # near-neutral (Z ~ 0) instead of being spuriously adverse (was Z ~ -1.54, which
    # inflated ECL coverage and Stage 2 share).
    ref_macro = merged[macro_cols].mean()

    # Through-the-cycle (unconditional) default rate — the baseline anchor.
    ttc_dr = float(np.clip(df_train["target"].mean(), 1e-4, 0.99))

    baseline_macro = {"const": 1.0, **{c: float(ref_macro[c]) for c in macro_cols}}

    # Recenter the projection intercept so the baseline (central) scenario reproduces the
    # TTC default rate exactly; upside/downside then move symmetrically around it. Keeps
    # dr = beta_adjusted . x intact (the report's scenario-DR QA identity still holds)
    # while removing the last-quarter bias that made "baseline" adverse.
    proj_params["const"] = ttc_dr - sum(
        proj_params.get(k, 0.0) * baseline_macro[k] for k in macro_cols
    )

    # CPI_inflation has no scenario delta (existing design): it stays at the
    # sample-mean baseline value in all three scenarios, same as before.
    upside_macro = baseline_macro.copy()
    for k, delta in _SCENARIO_DELTAS["upside"].items():
        if k in upside_macro:
            upside_macro[k] += delta
    if "UNRATE" in upside_macro:
        upside_macro["UNRATE"] = max(2.0, upside_macro["UNRATE"])
    if "FEDFUNDS" in upside_macro:
        upside_macro["FEDFUNDS"] = max(0.1, upside_macro["FEDFUNDS"])

    downside_macro = baseline_macro.copy()
    for k, delta in _SCENARIO_DELTAS["downside"].items():
        if k in downside_macro:
            downside_macro[k] += delta
    if "FEDFUNDS" in downside_macro:
        downside_macro["FEDFUNDS"] = max(0.1, downside_macro["FEDFUNDS"])

    # Predict default rates under each scenario using the sign-adjusted, recentred
    # coefficients (guarantees Downside > Baseline > Upside ordering, baseline = TTC).
    dr_base = np.clip(sum(proj_params.get(k, 0.0) * v for k, v in baseline_macro.items()), 1e-4, 0.99)
    dr_up = np.clip(sum(proj_params.get(k, 0.0) * v for k, v in upside_macro.items()), 1e-4, 0.99)
    dr_down = np.clip(sum(proj_params.get(k, 0.0) * v for k, v in downside_macro.items()), 1e-4, 0.99)

    # Map default rates to Vasicek shocks Z

    rho = 0.15  # supervisory correlation
    z_ttc = ndtri(ttc_dr)

    def get_z_shock(pit_dr):
        z_pit = ndtri(pit_dr)
        return (z_ttc - z_pit * np.sqrt(1.0 - rho)) / np.sqrt(rho)

    shocks = {
        "baseline": get_z_shock(dr_base),
        "upside": get_z_shock(dr_up),
        "downside": get_z_shock(dr_down),
        "elasticities": elasticities,               # raw OLS coefficients (reported honestly)
        "elasticities_adjusted": proj_params,       # sign-adjusted coefficients used for projection
        "macro_sign_adjusted": sign_adjusted,       # True if any coefficient sign was imposed
        "macro_unrate_lag": lag_used,               # quarters of macro lag actually applied
        "r_squared": float(model.rsquared),
        "predictions": {"baseline": dr_base, "upside": dr_up, "downside": dr_down},
        # Scenario input assumptions, exported for the report (Fix 1.3)
        "scenario_inputs": {
            "baseline": baseline_macro,
            "upside": upside_macro,
            "downside": downside_macro,
        },
    }
    logger.info("Macro-implied Vasicek shocks: baseline=%.4f | upside=%.4f | downside=%.4f",
                shocks["baseline"], shocks["upside"], shocks["downside"])

    return shocks
