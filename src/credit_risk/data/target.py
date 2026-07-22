"""Target variable definition for PD modelling.

Implements the bad/good/exclude mapping from config.
Excluded loans (Current, In Grace Period, Late 16-30 days) have no resolved
outcome and must be dropped before model training.
"""

from __future__ import annotations

import logging

import pandas as pd

from credit_risk.utils.config import TargetConfig

logger = logging.getLogger(__name__)

TARGET_COL = "target"
_BAD = 1
_GOOD = 0
_EXCLUDE = -1


def define_target(df: pd.DataFrame, cfg: TargetConfig) -> pd.DataFrame:
    """Add a binary 'target' column (1=bad, 0=good) and drop excluded rows.

    Parameters
    ----------
    df:
        DataFrame containing a 'loan_status' column.
    cfg:
        Target configuration (bad_statuses, good_statuses).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'target' column; excluded rows removed.
    """
    if "loan_status" not in df.columns:
        raise ValueError("Column 'loan_status' not found in DataFrame.")

    bad_set = set(cfg.bad_statuses)
    good_set = set(cfg.good_statuses)

    overlap = bad_set & good_set
    if overlap:
        raise ValueError(f"loan_status values appear in both bad and good sets: {overlap}")

    status = df["loan_status"]
    target = pd.Series(_EXCLUDE, index=df.index, dtype=int, name=TARGET_COL)
    target[status.isin(good_set)] = _GOOD
    target[status.isin(bad_set)] = _BAD

    n_bad = (target == _BAD).sum()
    n_good = (target == _GOOD).sum()
    n_excl = (target == _EXCLUDE).sum()

    logger.info(
        "Target definition: bad=%d (%.1f%%) | good=%d (%.1f%%) | excluded=%d (%.1f%%)",
        n_bad, n_bad / len(df) * 100,
        n_good, n_good / len(df) * 100,
        n_excl, n_excl / len(df) * 100,
    )

    unknown = set(status.dropna().unique()) - bad_set - good_set
    if unknown:
        logger.warning(
            "%d unique statuses not in bad/good sets (will be excluded): %s",
            len(unknown), sorted(str(v) for v in unknown),
        )

    out = df.copy()
    out[TARGET_COL] = target
    out = out[out[TARGET_COL] != _EXCLUDE].reset_index(drop=True)
    return out
