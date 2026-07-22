"""Time-series diagnostics that justify the macro model's lag and sign choices.

The production macro model (``risk/ifrs9_ecl.fit_macro_model``) imposes an economic sign
prior on the UNRATE coefficient because the raw contemporaneous OLS returns a spurious
negative sign. This module supplies the methodological backing for that decision, rather
than presenting it as an ad-hoc override:

* **ADF** stationarity tests on the default-rate and macro series;
* **Granger causality** of UNRATE on the default rate across candidate lags;
* **AIC** grid search over the UNRATE lag (with the expected economic signs verified);
* **Johansen** cointegration test and, if cointegrated, a **VECM** whose long-run
  relation is inspected for the correct UNRATE sign.

Everything is wrapped defensively: the merged quarterly series is short (~30-40 points),
so any test that cannot run degrades to ``None`` instead of failing the pipeline. This is
an *alongside* robustness analysis; it does not change the production scenario shocks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MACRO_COLS = ["UNRATE", "GDP_growth", "FEDFUNDS", "CPI_inflation", "HPI_growth"]


def build_quarterly_macro_frame(df_train: pd.DataFrame, macro_path: str) -> pd.DataFrame:
    """Assemble the quarterly (default_rate + macro) frame used for the diagnostics.

    Mirrors the contemporaneous merge inside ``fit_macro_model`` so the two analyses
    describe the same series.
    """
    macro_df = pd.read_csv(macro_path)
    d = df_train.copy()
    d["quarter"] = (
        pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce").dt.to_period("Q").astype(str)
    )
    q = d.groupby("quarter")["target"].agg(["count", "sum"])
    q["default_rate"] = q["sum"] / q["count"]
    q = q.reset_index()
    merged = q.merge(macro_df, on="quarter", how="inner").sort_values("quarter")
    return merged.reset_index(drop=True)


def _adf(series: pd.Series) -> dict[str, float | bool] | None:
    try:
        from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415

        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 8 or s.nunique() < 3:
            return None
        stat, pval = adfuller(s, autolag="AIC")[:2]
        return {"stat": float(stat), "pvalue": float(pval), "stationary": bool(pval < 0.05)}
    except Exception as err:  # noqa: BLE001
        logger.debug("ADF failed: %s", err)
        return None


def _granger(merged: pd.DataFrame, target: str, cause: str, max_lag: int) -> dict | None:
    """Test whether ``cause`` Granger-causes ``target`` on first differences."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests  # noqa: PLC0415

        data = merged[[target, cause]].apply(pd.to_numeric, errors="coerce").diff().dropna()
        usable = int(min(max_lag, max(1, len(data) // 5)))
        if len(data) < 10 or usable < 1:
            return None
        try:
            res = grangercausalitytests(data, maxlag=usable, verbose=False)
        except TypeError:  # newer statsmodels dropped the verbose kwarg
            res = grangercausalitytests(data, maxlag=usable)
        pvals = {str(lag): float(res[lag][0]["ssr_ftest"][1]) for lag in res}
        best_lag = int(min(pvals, key=pvals.get))
        min_pvalue = pvals[str(best_lag)]
        # Bonferroni-correct the significance threshold for testing `usable`
        # lags: thresholding the uncorrected min p-value at 0.10 gives an
        # effective false-positive rate of ~1-(1-0.10)^usable (~30% for
        # usable=4) on this short (~30-quarter) series, so noise alone
        # frequently produces a spurious "causal" verdict.
        alpha_corrected = 0.10 / usable
        return {
            "p_values": pvals,
            "best_lag": best_lag,
            "min_pvalue": min_pvalue,
            "alpha_corrected": alpha_corrected,
            "causal": bool(min_pvalue < alpha_corrected),
        }
    except Exception as err:  # noqa: BLE001
        logger.debug("Granger failed: %s", err)
        return None


def _aic_lag_selection(
    merged: pd.DataFrame, target: str, cols: list[str], max_lag: int
) -> dict | None:
    """Grid-search the UNRATE lag by AIC; report the fitted signs at the optimum.

    Every candidate lag is fit on the same (max-lag-truncated) sample.
    Shifting UNRATE by ``lag`` NaNs out the first ``lag`` rows, so a naive
    per-lag ``dropna()`` fits each lag on a different-sized sample; since
    AIC = 2k - 2*logL and logL scales with n, AIC is not comparable across
    sample sizes and the grid search is biased toward higher lags. Fitting
    every lag on the rows valid for the *largest* lag keeps n fixed.
    """
    try:
        import statsmodels.api as sm  # noqa: PLC0415

        present = [c for c in cols if c in merged.columns]
        if "UNRATE" not in present:
            return None
        if len(merged) <= max_lag + 3:
            return None
        common_idx = merged.index[max_lag:]
        best: dict | None = None
        for lag in range(0, max_lag + 1):
            d = merged.copy()
            d["UNRATE_lag"] = pd.to_numeric(d["UNRATE"], errors="coerce").shift(lag)
            regressors = ["UNRATE_lag"] + [c for c in present if c != "UNRATE"]
            data = (
                d.loc[common_idx, [target, *regressors]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
            if len(data) < len(regressors) + 3:
                continue
            model = sm.OLS(data[target], sm.add_constant(data[regressors])).fit()
            if best is None or model.aic < best["aic"]:
                best = {
                    "lag": lag,
                    "aic": float(model.aic),
                    "unrate_coef": float(model.params.get("UNRATE_lag", float("nan"))),
                    "gdp_coef": float(model.params.get("GDP_growth", float("nan"))),
                    "r_squared": float(model.rsquared),
                }
        if best is None:
            return None
        best["unrate_sign_ok"] = bool(best["unrate_coef"] > 0)
        best["gdp_sign_ok"] = bool(
            np.isnan(best["gdp_coef"]) or best["gdp_coef"] < 0
        )
        return best
    except Exception as err:  # noqa: BLE001
        logger.debug("AIC lag selection failed: %s", err)
        return None


def _johansen_vecm(merged: pd.DataFrame, target: str, cols: list[str]) -> dict | None:
    try:
        from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen  # noqa: PLC0415

        use = [target] + [c for c in ("UNRATE", "GDP_growth") if c in cols]
        data = merged[use].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 12:
            return None
        joh = coint_johansen(data, det_order=0, k_ar_diff=1)
        trace_stat = float(joh.lr1[0])
        crit_5 = float(joh.cvt[0, 1])
        cointegrated = bool(trace_stat > crit_5)
        out: dict[str, float | bool | None] = {
            "trace_stat": trace_stat,
            "crit_5pct": crit_5,
            "cointegrated": cointegrated,
            "vecm_unrate_sign": None,
        }
        if cointegrated and "UNRATE" in data.columns:
            vecm = VECM(data, k_ar_diff=1, coint_rank=1).fit()
            beta = np.asarray(vecm.beta).ravel()
            names = list(data.columns)
            if "UNRATE" in names:
                out["vecm_unrate_sign"] = float(np.sign(beta[names.index("UNRATE")]))
        return out
    except Exception as err:  # noqa: BLE001
        logger.debug("Johansen/VECM failed: %s", err)
        return None


def analyze_macro_timeseries(
    merged: pd.DataFrame,
    *,
    target: str = "default_rate",
    macro_cols: list[str] | None = None,
    max_lag: int = 4,
) -> dict:
    """Run the full ADF / Granger / AIC / Johansen-VECM diagnostic suite.

    Returns a dict with keys ``adf, granger, aic_lag_selection, johansen, n_quarters``;
    any sub-analysis that cannot run on the (short) series is ``None``.
    """
    cols = macro_cols or _MACRO_COLS
    present = [c for c in cols if c in merged.columns]
    adf = {"default_rate": _adf(merged[target])}
    if "UNRATE" in present:
        adf["UNRATE"] = _adf(merged["UNRATE"])

    result = {
        "n_quarters": int(len(merged)),
        "adf": adf,
        "granger": _granger(merged, target, "UNRATE", max_lag) if "UNRATE" in present else None,
        "aic_lag_selection": _aic_lag_selection(merged, target, present, max_lag),
        "johansen": _johansen_vecm(merged, target, present),
    }
    logger.info(
        "Macro TS diagnostics: n=%d | granger=%s | aic_lag=%s",
        result["n_quarters"],
        (result["granger"] or {}).get("causal"),
        (result["aic_lag_selection"] or {}).get("lag"),
    )
    return result
