"""Out-of-sample validation for the LGD model.

The production LGD model (``models/lgd.LGDModel``) is fitted on training-vintage defaults
but had no chronological out-of-sample validation. This module scores a held-out set of
defaulted loans and reports the standard LGD backtesting metrics — MAE, RMSE, R2, a
Kolmogorov-Smirnov distributional test, and a decile calibration table — mirroring the
severity-model validation expected in regulatory model documentation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error

from credit_risk.models.lgd import compute_realised_lgd

logger = logging.getLogger(__name__)


def _lgd_error_metrics(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float, float]:
    """(mae, rmse, r2, ks_stat, ks_pvalue) for aligned finite actual/pred arrays."""
    mae = float(mean_absolute_error(actual, pred))
    mse = float(mean_squared_error(actual, pred))
    rmse = float(np.sqrt(mse))
    var = float(np.var(actual))
    r2 = float(1.0 - mse / var) if var > 0 else float("nan")
    ks_stat, ks_pvalue = ks_2samp(actual, pred)
    return mae, rmse, r2, float(ks_stat), float(ks_pvalue)


def validate_lgd_models(
    model: object,
    df_test: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Validate the two-stage champion and the LightGBM challenger OOS, side by side.

    Returns ``{"champion": {...}, "challenger": {...}, "recommended": "champion"|"challenger"}``
    where each metric dict has ``mae, rmse, r2, ks_stat, n_test``. The recommendation is
    the model with the lower OOS RMSE (only when the challenger exists and strictly wins).
    """
    actual_full = compute_realised_lgd(df_test).to_numpy(dtype=float)

    def _score(pred_series) -> dict[str, float] | None:
        pred = np.asarray(pred_series, dtype=float)
        mask = np.isfinite(actual_full) & np.isfinite(pred)
        a, p = actual_full[mask], pred[mask]
        if a.size == 0:
            return None
        mae, rmse, r2, ks_stat, _ = _lgd_error_metrics(a, p)
        return {"mae": mae, "rmse": rmse, "r2": r2, "ks_stat": ks_stat, "n_test": float(a.size)}

    was = getattr(model, "_use_challenger", False)
    try:
        model._use_challenger = False  # force two-stage path for the champion score
        champ = _score(model.predict(df_test))  # type: ignore[attr-defined]
    finally:
        model._use_challenger = was

    out: dict[str, dict[str, float]] = {}
    if champ is not None:
        out["champion"] = champ
    chal = None
    if getattr(model, "_challenger", None) is not None:
        chal = _score(model.predict_challenger(df_test))  # type: ignore[attr-defined]
        if chal is not None:
            out["challenger"] = chal

    recommended = "champion"
    if champ is not None and chal is not None and chal["rmse"] < champ["rmse"]:
        recommended = "challenger"
    out["recommended"] = recommended  # type: ignore[assignment]
    logger.info(
        "LGD champion vs challenger OOS: champ RMSE=%.4f R2=%.4f | chal RMSE=%s -> recommend %s",
        (champ or {}).get("rmse", float("nan")), (champ or {}).get("r2", float("nan")),
        f"{chal['rmse']:.4f}" if chal else "n/a", recommended,
    )
    return out


def validate_lgd(
    model: object,
    df_test: pd.DataFrame,
    *,
    n_deciles: int = 10,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Validate a fitted LGD model on out-of-sample defaulted loans.

    Parameters
    ----------
    model:
        A fitted ``LGDModel`` exposing ``predict(df) -> pd.Series`` (name ``lgd_pred``).
    df_test:
        Held-out defaulted loans carrying the post-origination columns needed by
        ``compute_realised_lgd`` (``funded_amnt``, ``total_pymnt``, ``total_rec_prncp``).
    n_deciles:
        Number of predicted-LGD buckets for the calibration table.

    Returns
    -------
    (metrics, decile_df) where ``metrics`` has keys ``mae, rmse, r2, ks_stat, ks_pvalue,
    n_test`` and ``decile_df`` has columns ``decile, mean_predicted, mean_actual, count``.
    """
    actual = compute_realised_lgd(df_test).to_numpy(dtype=float)
    pred = np.asarray(model.predict(df_test), dtype=float)  # type: ignore[attr-defined]

    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    n_test = int(actual.size)
    if n_test == 0:
        empty = pd.DataFrame(columns=["decile", "mean_predicted", "mean_actual", "count"])
        return (
            {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
             "ks_stat": float("nan"), "ks_pvalue": float("nan"), "n_test": 0.0},
            empty,
        )

    mae, rmse, r2, ks_stat, ks_pvalue = _lgd_error_metrics(actual, pred)

    tbl = pd.DataFrame({"predicted": pred, "actual": actual})
    try:
        tbl["decile"] = pd.qcut(tbl["predicted"], n_deciles, labels=False, duplicates="drop")
    except ValueError:  # too few distinct predictions to bin
        tbl["decile"] = 0
    decile_df = (
        tbl.groupby("decile", observed=True)
        .agg(mean_predicted=("predicted", "mean"),
             mean_actual=("actual", "mean"),
             count=("actual", "size"))
        .reset_index()
    )

    metrics = {
        "mae": mae, "rmse": rmse, "r2": r2,
        "ks_stat": float(ks_stat), "ks_pvalue": float(ks_pvalue),
        "n_test": float(n_test),
    }
    logger.info(
        "LGD OOS validation (n=%d): MAE=%.4f | RMSE=%.4f | R2=%.4f | KS=%.4f (p=%.4f)",
        n_test, mae, rmse, r2, ks_stat, ks_pvalue,
    )
    return metrics, decile_df
