"""Out-of-time (OOT) train/validation/test splitting.

Banks require OOT validation — random splits hide temporal degradation.
Split by loan origination date (issue_d) not by random sampling.

    Train:         issue_d < train_cutoff
    In-time test:  random holdout_frac from the train period
    OOT:           issue_d >= oot_cutoff
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from credit_risk.utils.config import SplitConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/test/OOT splits."""

    train: pd.DataFrame
    test: pd.DataFrame    # in-time holdout (random subset of train period)
    oot: pd.DataFrame     # out-of-time (most recent vintages)
    full_accepted: pd.DataFrame | None = None  # pre-leakage-filtered data for LGD/EAD

    def __repr__(self) -> str:
        return (
            f"DataSplit(train={len(self.train):,}, "
            f"test={len(self.test):,}, "
            f"oot={len(self.oot):,})"
        )


def parse_issue_date(df: pd.DataFrame) -> pd.Series:
    """Parse Lending Club's 'MMM-YYYY' issue_d to datetime."""
    return pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")


def time_split(
    df: pd.DataFrame,
    cfg: SplitConfig,
    seed: int = 42,
) -> DataSplit:
    """Split loans into train / in-time test / OOT by origination date.

    Parameters
    ----------
    df:
        Cleaned DataFrame with 'issue_d' column.
    cfg:
        Split configuration (train_cutoff, oot_cutoff, holdout_frac).
    seed:
        Random seed for in-time holdout sampling.

    Returns
    -------
    DataSplit
        Named splits with no temporal overlap.
    """
    if "issue_d" not in df.columns:
        raise ValueError("Column 'issue_d' not found.")

    issue_dt = parse_issue_date(df)
    train_cutoff = pd.Timestamp(cfg.train_cutoff)
    oot_cutoff = pd.Timestamp(cfg.oot_cutoff)

    if train_cutoff >= oot_cutoff:
        raise ValueError(
            f"train_cutoff ({train_cutoff}) must be before oot_cutoff ({oot_cutoff})."
        )

    train_mask = issue_dt < train_cutoff
    oot_mask = issue_dt >= oot_cutoff
    # Between train_cutoff and oot_cutoff is "grey zone" — excluded (immature at OOT time)

    train_pool = df[train_mask].copy()
    oot = df[oot_mask].copy()

    # Random in-time holdout from train period
    test = train_pool.sample(frac=cfg.holdout_frac, random_state=seed)
    train = train_pool.drop(test.index)

    logger.info(
        "OOT split | cutoffs: train<%s, oot>=%s\n"
        "  train=%d  |  in-time test=%d  |  OOT=%d  |  grey-zone=%d",
        train_cutoff.date(), oot_cutoff.date(),
        len(train), len(test), len(oot),
        (df[~train_mask & ~oot_mask]).shape[0],
    )

    # Sanity: no date overlap
    _assert_no_overlap(train, test, oot, issue_dt)

    return DataSplit(train=train.reset_index(drop=True),
                     test=test.reset_index(drop=True),
                     oot=oot.reset_index(drop=True))


def _assert_no_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame,
    issue_dt: pd.Series,
) -> None:
    """Assert train/test share no rows with OOT."""
    train_test_idx = set(train.index) | set(test.index)
    oot_idx = set(oot.index)
    overlap = train_test_idx & oot_idx
    if overlap:
        raise RuntimeError(
            f"OOT split has {len(overlap)} rows overlapping with train/test. "
            "This is a data leakage bug."
        )
