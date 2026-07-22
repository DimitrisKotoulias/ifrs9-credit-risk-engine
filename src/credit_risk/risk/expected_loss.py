"""Expected Loss calculation.

EL = PD × LGD × EAD  (per loan)

Portfolio EL aggregated by grade, vintage, purpose, term.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_expected_loss(
    df: pd.DataFrame,
    pd_col: str = "pd_pred",
    lgd_col: str = "lgd_pred",
    ead_col: str = "ead",
) -> pd.Series:
    """Compute per-loan Expected Loss.

    Parameters
    ----------
    df:
        DataFrame with PD, LGD, EAD columns.
    pd_col, lgd_col, ead_col:
        Column names.

    Returns
    -------
    pd.Series
        Per-loan EL, indexed as df.
    """
    pd_arr = df[pd_col].clip(0.0, 1.0)
    lgd_arr = df[lgd_col].clip(0.0, 1.0)
    ead_arr = df[ead_col].clip(lower=0.0)

    el = pd_arr * lgd_arr * ead_arr
    logger.info(
        "Portfolio EL: total=%.2f | mean=%.2f | mean EL rate=%.4f%%",
        el.sum(), el.mean(), el.mean() / ead_arr.mean() * 100,
    )
    return el.rename("el")


def portfolio_el_summary(
    df: pd.DataFrame,
    el_col: str = "el",
    ead_col: str = "ead",
    pd_col: str = "pd_pred",
) -> dict[str, object]:
    """Compute portfolio-level EL summary metrics.

    Returns
    -------
    dict
        total_el, total_ead, el_rate, mean_pd, mean_lgd, n_loans.
    """
    total_el = float(df[el_col].sum())
    total_ead = float(df[ead_col].sum())
    el_rate = total_el / total_ead if total_ead > 0 else 0.0

    return {
        "total_el": total_el,
        "total_ead": total_ead,
        "el_rate": el_rate,
        "mean_pd": float(df[pd_col].mean()),
        "n_loans": len(df),
    }


def el_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    el_col: str = "el",
    ead_col: str = "ead",
) -> pd.DataFrame:
    """EL breakdown by a single segment dimension.

    Parameters
    ----------
    df:
        Portfolio DataFrame.
    segment_col:
        Column to group by (e.g. 'grade', 'purpose').
    el_col, ead_col:
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: segment, n_loans, total_ead, total_el, el_rate.
    """
    if segment_col not in df.columns:
        logger.warning("Segment column '%s' not found.", segment_col)
        return pd.DataFrame()

    grp = df.groupby(segment_col).agg(
        n_loans=(el_col, "count"),
        total_ead=(ead_col, "sum"),
        total_el=(el_col, "sum"),
    ).reset_index()
    grp["el_rate"] = grp["total_el"] / grp["total_ead"].clip(lower=1.0)
    grp = grp.rename(columns={segment_col: "segment"})
    return grp.sort_values("total_el", ascending=False).reset_index(drop=True)


def run_expected_loss(
    df: pd.DataFrame,
    pd_col: str = "pd_pred",
    lgd_col: str = "lgd_pred",
    ead_col: str = "ead",
) -> pd.DataFrame:
    """Full EL computation: add el column and compute breakdowns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'el' column added.
    """
    out = df.copy()
    out["el"] = compute_expected_loss(out, pd_col, lgd_col, ead_col)

    summary = portfolio_el_summary(out)
    logger.info("Portfolio EL summary: %s", summary)

    breakdowns = {}
    for seg in ["grade", "purpose", "term"]:
        if seg in out.columns:
            breakdowns[seg] = el_by_segment(out, seg)

    out.attrs["el_summary"] = summary
    # Convert breakdown DataFrames to dicts for JSON serialisability (parquet metadata)
    out.attrs["el_breakdowns"] = {
        k: v.to_dict(orient="records") for k, v in breakdowns.items()
    }

    return out
