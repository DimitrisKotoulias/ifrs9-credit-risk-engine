"""Concentration risk: Herfindahl-Hirschman Index and the Granularity Adjustment.

The Basel ASRF model assumes an infinitely-granular, perfectly-diversified portfolio.
Real portfolios carry name and segment concentration that the single-factor model
ignores, so a capital add-on is required. This module measures concentration with the
Herfindahl-Hirschman Index (HHI) across exposure dimensions and quantifies the extra
capital via a simplified Gordy-Lutkebohmert granularity adjustment.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PD_CLIP = 1e-9


def herfindahl_index(exposures: np.ndarray | pd.Series) -> float:
    """Herfindahl-Hirschman Index = sum of squared exposure shares, in (0, 1].

    HHI = 1 for a single name; HHI = 1/N for N equal exposures.
    """
    e = np.asarray(exposures, dtype=float)
    e = e[e > 0]
    total = e.sum()
    if total <= 0 or e.size == 0:
        return 0.0
    shares = e / total
    return float((shares ** 2).sum())


def effective_n(hhi: float) -> float:
    """Effective number of equal-sized exposures implied by an HHI."""
    return float(1.0 / hhi) if hhi > 0 else float("inf")


def granularity_adjustment(
    pd_arr: np.ndarray,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    rho: float = 0.15,  # noqa: ARG001 - kept for API symmetry / future ASRF variants
) -> float:
    """Simplified Gordy-Lutkebohmert granularity adjustment (capital surcharge).

    Uses the idiosyncratic unexpected-loss volatility per exposure,
    ``UL_i = sqrt(PD_i (1-PD_i)) * LGD_i * EAD_i``, and returns the concentration
    add-on ``GA = sum(UL_i^2) / (2 * total_EAD)``. GA -> 0 as the portfolio becomes
    granular (many small exposures) and grows with name concentration.
    """
    pd_arr = np.clip(np.asarray(pd_arr, dtype=float), _PD_CLIP, 1.0 - _PD_CLIP)
    lgd_arr = np.clip(np.asarray(lgd_arr, dtype=float), 0.0, 1.0)
    ead_arr = np.asarray(ead_arr, dtype=float)
    total_ead = float(ead_arr.sum())
    if total_ead <= 0:
        return 0.0
    ul = np.sqrt(pd_arr * (1.0 - pd_arr)) * lgd_arr * ead_arr
    return float((ul ** 2).sum() / (2.0 * total_ead))


def hhi_by_dimension(
    df: pd.DataFrame,
    dims: list[str],
    exposure_col: str = "funded_amnt",
) -> pd.DataFrame:
    """HHI, effective N, category count and top-category share per exposure dimension."""
    rows = []
    for dim in dims:
        if dim not in df.columns:
            continue
        grouped = df.groupby(dim, observed=True)[exposure_col].sum()
        grouped = grouped[grouped > 0]
        if grouped.empty:
            continue
        hhi = herfindahl_index(grouped.to_numpy())
        rows.append({
            "dimension": dim,
            "hhi": hhi,
            "effective_n": effective_n(hhi),
            "n_categories": int(grouped.size),
            "top_share": float(grouped.max() / grouped.sum()),
        })
    return pd.DataFrame(rows)


def grouped_exposures(
    df: pd.DataFrame,
    dims: list[str],
    exposure_col: str = "funded_amnt",
) -> dict[str, pd.Series]:
    """Exposure totals per category for each dimension (for plotting)."""
    out: dict[str, pd.Series] = {}
    for dim in dims:
        if dim in df.columns:
            s = df.groupby(dim, observed=True)[exposure_col].sum()
            out[dim] = s[s > 0].sort_values(ascending=False)
    return out


def run_concentration(
    df: pd.DataFrame,
    *,
    dims: list[str] | None = None,
    exposure_col: str = "funded_amnt",
    pd_col: str = "pd_pred",
    lgd_col: str = "lgd_pred",
    ead_col: str = "ead",
    rho: float = 0.15,
) -> tuple[dict[str, object], dict[str, pd.Series]]:
    """Compute the concentration summary and per-dimension grouped exposures.

    Returns ``(summary, grouped)`` where ``summary`` has keys ``dimensions``
    (list of dicts) and ``granularity_adjustment``.
    """
    dims = dims or ["grade", "purpose", "addr_state"]
    exposure = exposure_col if exposure_col in df.columns else ead_col
    hhi_df = hhi_by_dimension(df, dims, exposure)
    grouped = grouped_exposures(df, dims, exposure)
    ga = granularity_adjustment(
        df[pd_col].to_numpy(dtype=float),
        df[lgd_col].to_numpy(dtype=float),
        df[ead_col].to_numpy(dtype=float),
        rho=rho,
    )
    summary: dict[str, object] = {
        "dimensions": hhi_df.to_dict(orient="records"),
        "granularity_adjustment": ga,
        "exposure_col": exposure,
    }
    logger.info(
        "Concentration: %s | GA surcharge=%.0f",
        {r["dimension"]: round(r["hhi"], 4) for r in summary["dimensions"]}, ga,
    )
    return summary, grouped
