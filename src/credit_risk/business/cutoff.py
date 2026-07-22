"""Credit cut-off optimisation.

Sweeps score thresholds, computes approval rate / bad rate / expected profit,
and finds the optimal cut-off that maximises expected profit.

Cost matrix (configurable):
    profit_good  = revenue earned on a good loan (interest income)
    loss_bad     = loss on a bad loan (LGD × EAD)
    cost_decline = opportunity cost of declining a good loan (≈0 by default)

Expected profit at threshold s:
    π(s) = approved_good × profit_good − approved_bad × loss_bad − declined_good × cost_decline
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sweep_cutoffs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ead: np.ndarray | None = None,
    profit_good: float = 0.05,
    loss_bad: float = 0.45,
    cost_decline: float = 0.0,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """Sweep score thresholds and compute business metrics.

    Parameters
    ----------
    y_true:
        True default labels (1=bad, 0=good).
    y_score:
        Model score (higher = lower risk / approve).
    ead:
        EAD per loan. If None, uses unit exposure.
    profit_good:
        Profit earned per unit EAD on approved goods (fraction).
    loss_bad:
        Loss per unit EAD on approved bads (fraction, = LGD proxy).
    cost_decline:
        Cost per unit EAD on declined goods (opportunity cost).
    n_thresholds:
        Number of threshold points to sweep.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, n_approved, approval_rate, bad_rate, expected_profit,
        profit_per_ead, gini_approved, approved_good, approved_bad, declined_bad,
        declined_good.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    ead = np.ones(len(y_true)) if ead is None else np.asarray(ead, dtype=float)

    thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)
    records = []

    for thr in thresholds:
        approved = y_score >= thr
        n_approved = approved.sum()
        n_total = len(y_true)

        if n_approved == 0:
            records.append({
                "threshold": thr, "n_approved": 0, "approval_rate": 0.0,
                "bad_rate": 0.0, "expected_profit": 0.0, "profit_per_ead": 0.0,
                "approved_good": 0, "approved_bad": 0,
                "declined_good": int((~approved & (y_true == 0)).sum()),
                "declined_bad": int((~approved & (y_true == 1)).sum()),
            })
            continue

        y_app = y_true[approved]
        ead_app = ead[approved]

        approved_good = int((approved & (y_true == 0)).sum())
        approved_bad = int((approved & (y_true == 1)).sum())
        declined_good = int((~approved & (y_true == 0)).sum())
        declined_bad = int((~approved & (y_true == 1)).sum())

        # EAD-weighted profit
        ead_goods = ead[approved & (y_true == 0)]
        ead_bads = ead[approved & (y_true == 1)]
        ead_declined_goods = ead[~approved & (y_true == 0)]

        profit = (
            ead_goods.sum() * profit_good
            - ead_bads.sum() * loss_bad
            - ead_declined_goods.sum() * cost_decline
        )
        total_ead_app = ead_app.sum()

        records.append({
            "threshold": thr,
            "n_approved": int(n_approved),
            "approval_rate": n_approved / n_total,
            "bad_rate": float(y_app.mean()) if len(y_app) > 0 else 0.0,
            "expected_profit": float(profit),
            "profit_per_ead": float(profit / total_ead_app) if total_ead_app > 0 else 0.0,
            "approved_good": approved_good,
            "approved_bad": approved_bad,
            "declined_good": declined_good,
            "declined_bad": declined_bad,
        })

    return pd.DataFrame(records)


def optimal_cutoff(sweep_df: pd.DataFrame) -> dict[str, object]:
    """Find optimal threshold maximising expected profit.

    Returns
    -------
    dict with threshold, approval_rate, bad_rate, expected_profit, row index.
    """
    idx = int(sweep_df["expected_profit"].idxmax())
    row = sweep_df.iloc[idx]
    result = {
        "threshold": float(row["threshold"]),
        "approval_rate": float(row["approval_rate"]),
        "bad_rate": float(row["bad_rate"]),
        "expected_profit": float(row["expected_profit"]),
        "n_approved": int(row["n_approved"]),
        "sweep_idx": idx,
    }
    logger.info(
        "Optimal cutoff: threshold=%.4f | approval=%.1f%% | bad_rate=%.3f%% | profit=%.2f",
        result["threshold"],
        result["approval_rate"] * 100,
        result["bad_rate"] * 100,
        result["expected_profit"],
    )
    return result


def raroc_argmax_cutoff(cutoff_strategy: list[dict]) -> dict | None:
    """Cutoff that maximises portfolio RAROC over the swept grid.

    On a risk-priced book this is typically the most inclusive cutoff (higher-risk
    grades carry high enough interest to stay RAROC-accretive), i.e. a corner. It
    is reported for context alongside the risk-appetite operating cutoff.
    """
    eligible = [r for r in cutoff_strategy if r.get("approval_rate", 0.0) > 0.0]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r.get("raroc", 0.0))


def risk_appetite_cutoff(
    cutoff_strategy: list[dict],
    max_bad_rate: float,
    min_approval_rate: float = 0.0,
) -> dict | None:
    """Recommended operating cutoff under a board bad-rate risk-appetite ceiling.

    Unconstrained profit/RAROC maximisation on a high-yield book approves the
    entire population (a corner). The operational cutoff is instead the most
    inclusive threshold (lowest score, hence highest approved volume and profit)
    whose approved-population bad rate stays within the risk-appetite ceiling ---
    i.e. profit maximisation subject to ``bad_rate <= max_bad_rate``. Because the
    approved bad rate rises monotonically as the cutoff is lowered, this yields a
    well-defined interior cutoff.

    A ``min_approval_rate`` floor guards against the opposite degenerate corner: if
    the bad-rate ceiling is so tight that it is only met at a near-zero approval
    rate, the reported operating point would be economically vacuous. When no cutoff
    satisfies the ceiling at the required volume, we fall back to the lowest-bad-rate
    cutoff that still approves at least the floor --- a genuine interior operating
    point rather than a ~0% corner.

    Parameters
    ----------
    cutoff_strategy:
        Rows (dicts) with keys ``cutoff``, ``approval_rate``, ``bad_rate`` ---
        cumulative for "approve all with score >= cutoff".
    max_bad_rate:
        Board risk-appetite ceiling on the approved-population bad rate.
    min_approval_rate:
        Minimum approved-volume fraction the operating point must achieve
        (default 0.0 preserves the original ceiling-only behaviour).

    Returns
    -------
    dict | None
        The chosen cumulative row, or None if no cutoff meets the constraints.
    """
    floor = max(min_approval_rate, 1e-9)
    eligible = [
        r for r in cutoff_strategy
        if r.get("approval_rate", 0.0) >= floor and r.get("bad_rate", 1.0) <= max_bad_rate
    ]
    if eligible:
        return min(eligible, key=lambda r: r["cutoff"])  # most inclusive within appetite
    if min_approval_rate > 0.0:
        # Ceiling unmet at the required volume: pick the best-risk cutoff meeting the floor
        # so the headline is a real interior operating point, not a vacuous ~0% corner.
        floor_rows = [r for r in cutoff_strategy if r.get("approval_rate", 0.0) >= floor]
        if floor_rows:
            return min(floor_rows, key=lambda r: r.get("bad_rate", 1.0))
    return None


def run_cutoff_analysis(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "target",
    ead_col: str = "ead",
    profit_good: float = 0.05,
    loss_bad: float = 0.45,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Full cut-off sweep + optimisation.

    Returns
    -------
    (sweep_df, optimal_dict)
    """
    y_true = df[target_col].values
    y_score = df[score_col].values
    ead = df[ead_col].values if ead_col in df.columns else None

    sweep_df = sweep_cutoffs(y_true, y_score, ead, profit_good, loss_bad)
    opt = optimal_cutoff(sweep_df)
    return sweep_df, opt
