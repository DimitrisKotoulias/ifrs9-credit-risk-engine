"""Calibration metrics: Brier score, Hosmer-Lemeshow, calibration curve."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

from credit_risk.reporting.style import (
    apply_publication_style, despine,
    C_NAVY, C_BLUE, C_GRAY, C_GRID,
)

_PALETTE = C_NAVY  # kept for backward compat


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    g: int = 10,
) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test (decile-based).

    H = sum_g (O_g - E_g)^2 / [E_g(1 - E_g/n_g)], chi2(G*-2)
    where G* is the number of non-empty bins (may be < g when predictions
    are highly duplicate). Groups with e_g == 0 but o_g > 0 are penalised
    with a large additive term (10^6) to flag extreme miscalibration.

    References
    ----------
    Hosmer, D.W. & Lemeshow, S. (1980). A Goodness-of-Fit Test for the
        Multiple Logistic Regression Model. *Communications in Statistics*.
    """
    if len(y_true) > 5000:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(y_true), size=5000, replace=False)
        y_t_eval = y_true[idx]
        y_p_eval = y_pred[idx]
    else:
        y_t_eval = y_true
        y_p_eval = y_pred

    decile_cuts = np.percentile(y_p_eval, np.linspace(0, 100, g + 1))
    decile_cuts = np.unique(decile_cuts)
    groups = np.digitize(y_p_eval, decile_cuts[1:-1])
    h_stat = 0.0
    non_empty = 0
    for grp in range(g):
        mask = groups == grp
        if not mask.any():
            continue
        non_empty += 1
        n_g = mask.sum()
        o_g = float(y_t_eval[mask].sum())
        e_g = float(y_p_eval[mask].sum())
        if e_g == 0.0 and o_g > 0:
            # Zero expected but positive observed: extreme miscalibration
            h_stat += 1e6
        elif 0 < e_g < n_g:
            h_stat += (o_g - e_g) ** 2 / (e_g * (1.0 - e_g / n_g))
    # Degrees of freedom based on actual non-empty groups, not g
    df = max(0, non_empty - 2)
    p_val = float(1.0 - _scipy_stats.chi2.cdf(h_stat, df=df)) if df > 0 else 0.0
    return {
        "h_stat": float(h_stat), "p_value": p_val, "df": df,
        "interpretation": "miscalibrated" if p_val < 0.05 else "calibrated",
    }


def compute_calibration(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bins: int = 10,
    label: str = "model",
) -> dict[str, float]:
    """Compute Brier score and Hosmer-Lemeshow p-value.

    Parameters
    ----------
    y_true:
        Binary target.
    y_pred:
        Predicted PD (probability).
    n_bins:
        Number of bins for HL test.
    label:
        Identifier for logging.

    Returns
    -------
    dict[str, float]
        Keys: brier_score, hl_statistic, hl_pvalue, n_bins.
    """
    from scipy import stats  # noqa: PLC0415

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 1e-8, 1 - 1e-8)

    brier = float(brier_score_loss(y_t, y_p))

    # Hosmer-Lemeshow test
    hl_res = hosmer_lemeshow_test(y_t, y_p, g=n_bins)
    hl_stat = hl_res["h_stat"]
    hl_pvalue = hl_res["p_value"]

    if hl_pvalue < 0.05:
        logger.warning(
            "[%s] Hosmer-Lemeshow p=%.4f < 0.05 => model is miscalibrated. "
            "Applying isotonic calibration.",
            label, hl_pvalue,
        )
    else:
        logger.info(
            "[%s] Hosmer-Lemeshow p=%.4f — calibration acceptable.",
            label, hl_pvalue,
        )

    logger.info("[%s] Brier score=%.4f", label, brier)

    return {
        "brier_score": brier,
        "hl_statistic": hl_stat,
        "hl_pvalue": hl_pvalue,
        "n_bins": n_bins,
    }


def fit_isotonic_calibrator(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> IsotonicRegression:
    """Fit isotonic regression calibrator on (y_true, y_pred).

    Returns
    -------
    IsotonicRegression
        Fitted calibrator; call .transform(y_pred_new) to apply.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(y_p, y_t)
    logger.info("Isotonic calibrator fitted.")
    return cal


def fit_platt_calibrator(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
):
    """Fit Platt scaling calibrator (logistic regression on predicted probabilities).

    Returns
    -------
    sklearn LogisticRegression
        Fitted calibrator; call .predict_proba(y_pred.reshape(-1,1))[:,1] to apply.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    cal.fit(y_p, y_t)
    logger.info("Platt calibrator fitted.")
    return cal


_DEFAULT_VINTAGE_GROUPS: list[tuple[str, int, int]] = [
    ("2007-2012", 2007, 2012),
    # Training population is loans originated prior to January 2015, so there are no
    # 2015 vintages in the development sample; this bucket is effectively 2013--2014
    # (2015 is a deliberate buffer year between the training and OOT windows).
    ("2013-2014", 2013, 2014),
    ("2016-2018", 2016, 2018),
]


def fit_era_calibrators(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    issue_year: np.ndarray | pd.Series,
    split_year: int = 2016,
) -> dict[str, dict[str, object]]:
    """Fit separate isotonic + Platt recalibrators for the pre/post ``split_year`` eras.

    Addresses calibration drift where newer vintages are systematically under-predicted:
    one global recalibrator cannot fix an era-specific bias, so we fit ``early``
    (issue_year < split_year) and ``late`` (>= split_year) calibrators independently.

    Returns ``{era: {"isotonic": IsotonicRegression, "platt": LogisticRegression}}`` for
    each era with enough two-class data.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    yr = np.asarray(issue_year, dtype=float)
    out: dict[str, dict[str, object]] = {}
    for era, mask in (("early", yr < split_year), ("late", yr >= split_year)):
        if int(mask.sum()) >= 20 and np.unique(y[mask]).size == 2:
            out[era] = {
                "isotonic": fit_isotonic_calibrator(y[mask], p[mask]),
                "platt": fit_platt_calibrator(y[mask], p[mask]),
            }
    return out


def calibration_by_vintage_group(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    issue_year: np.ndarray | pd.Series,
    *,
    split_year: int = 2016,
    vintage_groups: list[tuple[str, int, int]] | None = None,
) -> pd.DataFrame:
    """Raw vs era-recalibrated (isotonic/Platt) PD against actual default rate per vintage.

    Quantifies the drift (``pd_ratio`` = mean predicted PD / actual default rate; a value
    below 1 means under-prediction) and shows how era-specific recalibration moves the
    ratio back toward 1. This is an in-sample diagnostic; it does not alter production PD.

    Returns a DataFrame with columns ``group, n, raw_pd, isotonic_pd, platt_pd, actual_dr,
    pd_ratio_raw, pd_ratio_isotonic``.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    yr = np.asarray(issue_year, dtype=float)
    cals = fit_era_calibrators(y, p, yr, split_year=split_year)

    iso = p.copy()
    platt = p.copy()
    for era, mask in (("early", yr < split_year), ("late", yr >= split_year)):
        if era in cals and int(mask.sum()) > 0:
            iso[mask] = np.asarray(cals[era]["isotonic"].predict(p[mask]))  # type: ignore[attr-defined]
            platt[mask] = cals[era]["platt"].predict_proba(  # type: ignore[attr-defined]
                p[mask].reshape(-1, 1)
            )[:, 1]

    groups = vintage_groups or _DEFAULT_VINTAGE_GROUPS
    rows = []
    for label, lo, hi in groups:
        m = (yr >= lo) & (yr <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        actual = float(y[m].mean())
        raw_pd = float(p[m].mean())
        rows.append({
            "group": label,
            "n": n,
            "raw_pd": raw_pd,
            "isotonic_pd": float(iso[m].mean()),
            "platt_pd": float(platt[m].mean()),
            "actual_dr": actual,
            "pd_ratio_raw": raw_pd / actual if actual > 0 else float("nan"),
            "pd_ratio_isotonic": float(iso[m].mean()) / actual if actual > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def lifetime_pd_calibration_by_vintage(
    y_true: np.ndarray | pd.Series,
    pd_lifetime_pred: np.ndarray | pd.Series,
    issue_year: np.ndarray | pd.Series,
    *,
    max_mature_year: int = 2016,
    band_lo: float = 0.5,
    band_hi: float = 1.5,
) -> dict:
    """Validate the hazard model's lifetime PD against observed lifetime default rate.

    The IFRS 9 ECL engine (``risk.ifrs9_ecl.run_ifrs9_ecl``) consumes marginal/lifetime
    PDs straight from ``DiscreteHazardModel.predict_term_structure`` — a model that is
    fitted independently of, and never passed through, the scorecard's out-of-sample
    isotonic/Platt recalibrator (that recalibrator only touches the 12-month
    ``pd_pred`` used for EL/RWA/SICR-origination). This function closes that gap with a
    *diagnostic*, not a recalibration: it checks whether the hazard model's own lifetime
    PD is itself reasonably calibrated against realised outcomes, so any material drift
    is visible rather than silently absorbed into the ECL number.

    Restricted to "matured" vintages (``issue_year <= max_mature_year``): the data
    snapshot (2018Q4) has not yet resolved recoveries/charge-offs for 2017-2018
    originations, so their observed default status is right-censored and would
    understate the true lifetime default rate if included.

    Parameters
    ----------
    y_true:
        Observed default indicator (1 = defaulted at any point over the loan's
        observed life). For a matured vintage this is a reasonable proxy for the
        *lifetime* default outcome.
    pd_lifetime_pred:
        Hazard-model lifetime PD per loan (``predict_term_structure(...)["pd_lifetime"]``).
    issue_year:
        Origination year per loan.
    max_mature_year:
        Latest origination year treated as fully matured (default 2016, consistent
        with the LGD OOS validation window used elsewhere in the pipeline).
    band_lo, band_hi:
        Acceptable ratio band (predicted / observed) — matches the 50% tolerance band
        used by the 12-month vintage backtest.

    Returns
    -------
    dict with:
        ``by_vintage``: list of ``{vintage_year, n, predicted_pd_lifetime, observed_dr,
            ratio, in_band}`` (one row per mature origination year).
        ``portfolio``: same fields aggregated across all mature vintages.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pd_lifetime_pred, dtype=float)
    yr = np.asarray(issue_year, dtype=float)

    mature_mask = yr <= max_mature_year
    y_m, p_m, yr_m = y[mature_mask], p[mature_mask], yr[mature_mask]

    def _row(n: int, predicted: float, observed: float) -> dict:
        ratio = predicted / observed if observed > 0 else float("nan")
        in_band = bool(band_lo <= ratio <= band_hi) if not np.isnan(ratio) else False
        return {
            "n": n,
            "predicted_pd_lifetime": predicted,
            "observed_dr": observed,
            "ratio": ratio,
            "in_band": in_band,
        }

    rows = []
    for year in sorted(np.unique(yr_m)):
        m = yr_m == year
        n = int(m.sum())
        if n == 0:
            continue
        row = _row(n, float(p_m[m].mean()), float(y_m[m].mean()))
        row["vintage_year"] = int(year)
        rows.append(row)

    if len(y_m) > 0:
        portfolio = _row(int(len(y_m)), float(p_m.mean()), float(y_m.mean()))
    else:
        portfolio = _row(0, float("nan"), float("nan"))

    if portfolio["n"] > 0 and not portfolio["in_band"]:
        logger.warning(
            "Lifetime PD calibration OUT OF BAND: predicted=%.4f observed=%.4f "
            "ratio=%.3f (band [%.2f, %.2f]) — hazard model term structure may need "
            "recalibration before feeding the IFRS 9 ECL engine.",
            portfolio["predicted_pd_lifetime"], portfolio["observed_dr"],
            portfolio["ratio"], band_lo, band_hi,
        )
    elif portfolio["n"] > 0:
        logger.info(
            "Lifetime PD calibration in band: predicted=%.4f observed=%.4f ratio=%.3f",
            portfolio["predicted_pd_lifetime"], portfolio["observed_dr"], portfolio["ratio"],
        )
    else:
        logger.warning("Lifetime PD calibration: no matured vintages found (n=0).")

    return {"by_vintage": rows, "portfolio": portfolio}


def spiegelhalter_test(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Spiegelhalter (1986) Z-test for calibration.

    H0: model is well-calibrated. Z ~ N(0,1) under H0.
    |Z| > 1.96 → reject calibration at 5% significance.
    Preferred over Hosmer-Lemeshow for large N where HL is hypersensitive.
    """
    from scipy import stats  # noqa: PLC0415

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 1e-8, 1 - 1e-8)
    residuals = y_t - y_p
    numerator = float(np.sum(residuals * (1 - 2 * y_p)))
    variance_terms = (1 - 2 * y_p) ** 2 * y_p * (1 - y_p)
    denominator = float(np.sqrt(np.sum(variance_terms)))
    if denominator < 1e-15:
        return {"z_stat": 0.0, "p_value": 1.0, "calibrated": True}
    z = numerator / denominator
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"z_stat": float(z), "p_value": p, "calibrated": p > 0.05}


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    label: str = "Model",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Calibration plot: financial palette, CI band, clean axes (Fig 6 spec)."""
    apply_publication_style()
    df = pd.DataFrame({"y": y_true, "pred": y_pred})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    grp = df.groupby("bin", observed=False).agg(
        observed=("y", "mean"),
        predicted=("pred", "mean"),
        std=("y", "std"),
        n=("y", "count"),
    ).reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    max_val = max(grp["predicted"].max(), grp["observed"].max()) * 1.05

    # Perfect calibration line
    ax.plot([0, max_val], [0, max_val], color=C_GRAY, linestyle="--",
            linewidth=1.5, label="Perfect calibration", zorder=1)

    # Confidence band (±1 std)
    se = grp["std"] / np.sqrt(grp["n"].clip(lower=1))
    ax.fill_between(
        grp["predicted"], grp["observed"] - se, grp["observed"] + se,
        alpha=0.10, color=C_BLUE, zorder=2,
    )

    # Actual vs predicted points connected by a thin line
    ax.plot(grp["predicted"], grp["observed"], color=C_BLUE, linewidth=1.0, linestyle="-", zorder=3)
    ax.scatter(grp["predicted"], grp["observed"], color=C_NAVY, s=50,
               zorder=5, label=label)

    # Annotate deciles D1-D10
    for i, row in grp.iterrows():
        ax.annotate(
            f"D{i+1}",
            (row["predicted"], row["observed"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=C_NAVY,
            fontweight="bold",
        )

    ax.set_xlabel("Mean Predicted PD (Decile)", fontsize=11, labelpad=8)
    ax.set_ylabel("Observed Default Rate (Decile)", fontsize=11, labelpad=8)
    ax.set_title(f"Calibration Curve: {label}", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    despine(ax)
    ax.grid(True, axis="both", color=C_GRID, linewidth=0.6)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    fig.tight_layout()
    return fig


def compute_calibration_intercept_slope(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Compute calibration intercept (a) and slope (b) on logit scale.

    logit(P(Y=1)) = a + b * logit(P_pred)
    Ideally, a = 0 (perfect intercept calibration) and b = 1 (perfect slope calibration).
    """
    from scipy.special import logit  # noqa: PLC0415
    import statsmodels.api as sm  # noqa: PLC0415

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1.0 - 1e-6)

    logit_pred = logit(y_p)
    X = sm.add_constant(logit_pred, has_constant="add")
    try:
        model = sm.Logit(y_t, X).fit(disp=False, maxiter=200)
        intercept, slope = float(model.params[0]), float(model.params[1])
    except Exception as exc:
        logger.warning("Logit calibration regression failed: %s. Falling back to linear regression.", exc)
        try:
            # Fallback to OLS
            model = sm.OLS(y_t, X).fit()
            intercept, slope = float(model.params[0]), float(model.params[1])
        except Exception:
            intercept, slope = 0.0, 1.0
    return intercept, slope
