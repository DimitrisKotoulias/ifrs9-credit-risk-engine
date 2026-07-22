"""Data leakage filter.

The PD model must use only information available at loan origination.
This module removes post-origination features based on an explicit deny-list
in config.yaml.

Leakage is the single most common fatal flaw in credit-risk portfolio projects.
Calling it out explicitly and handling it correctly is a key signal of competence.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from credit_risk.utils.config import LeakageConfig

logger = logging.getLogger(__name__)


def filter_origination_features(
    df: pd.DataFrame,
    cfg: LeakageConfig,
) -> pd.DataFrame:
    """Remove post-origination features from a DataFrame.

    Columns are dropped if they match any name in deny_list exactly, OR match
    a deny_list entry as a prefix pattern (e.g. 'total_pymnt' matches
    'total_pymnt_inv', 'total_pymnt_amnt', etc.).

    Parameters
    ----------
    df:
        Input DataFrame (may contain post-origination columns).
    cfg:
        Leakage configuration with deny_list and allow_overrides.

    Returns
    -------
    pd.DataFrame
        DataFrame with leakage columns removed.
    """
    deny_set = set(cfg.deny_list)
    allow_set = set(cfg.allow_overrides)

    # Build regex: match exact name OR name as prefix followed by non-word char
    pattern = "|".join(
        r"(?:^" + re.escape(d) + r"(?:_.*)?$)"
        for d in deny_set
    )
    regex = re.compile(pattern, re.IGNORECASE)

    cols_to_drop = [
        col for col in df.columns
        if regex.match(col) and col not in allow_set
    ]

    if cols_to_drop:
        logger.info(
            "Leakage filter dropping %d post-origination columns: %s",
            len(cols_to_drop), sorted(cols_to_drop),
        )

    return df.drop(columns=cols_to_drop)


def log_leakage_policy(cfg: LeakageConfig) -> None:
    """Log the current leakage deny-list for audit trail."""
    logger.info(
        "Leakage policy: %d denied columns, %d allow-overrides.\n"
        "Denied: %s",
        len(cfg.deny_list),
        len(cfg.allow_overrides),
        sorted(cfg.deny_list),
    )
