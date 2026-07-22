"""WoE transformer with IV computation.

Implements a scikit-learn compatible WoE transformer that wraps the binner
and provides the information value table for feature selection.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.features.binning import ManualMonotonicBinner, get_binner

logger = logging.getLogger(__name__)


class WoETransformer(BaseEstimator, TransformerMixin):
    """Transform numeric features into their WoE-encoded counterparts.

    Wraps the best available binner (optbinning or manual fallback).
    Non-numeric and non-binned columns are dropped (WoE model uses WoE features only).

    Parameters
    ----------
    variables:
        Subset of columns to bin and WoE-transform.
        If None, all numeric columns are used.
    max_n_bins:
        Maximum number of bins (ignored for optbinning fallback variant).
    min_bin_frac:
        Minimum fraction of observations per bin.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        max_n_bins: int = 10,
        min_bin_frac: float = 0.05,
    ) -> None:
        self.variables = variables
        self.max_n_bins = max_n_bins
        self.min_bin_frac = min_bin_frac
        self._binner: ManualMonotonicBinner | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoETransformer":
        self._binner = get_binner(
            variables=self.variables,
            max_n_bins=self.max_n_bins,
            min_bin_frac=self.min_bin_frac,
        )
        self._binner.fit(X, y)
        self.variables_ = self._binner.variables_
        self.iv_table_ = self._binner.get_iv_table()
        logger.info(
            "WoETransformer fitted on %d features. Top IVs:\n%s",
            len(self.variables_),
            self.iv_table_.sort_values("iv", ascending=False).head(10).to_string(index=False),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._binner is None:
            raise RuntimeError("Call fit() before transform().")
        return self._binner.transform(X)[self.variables_]

    def get_iv_table(self) -> pd.DataFrame:
        """Return IV table sorted descending by IV."""
        if self._binner is None:
            raise RuntimeError("Call fit() first.")
        return self.iv_table_.sort_values("iv", ascending=False).reset_index(drop=True)


def compute_woe_iv(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    laplace_alpha: float = 0.5,
) -> tuple[pd.DataFrame, float]:
    """Compute WoE and IV for a single feature.

    Parameters
    ----------
    x:
        Feature series (numeric or categorical).
    y:
        Binary target (0/1).
    n_bins:
        Number of quantile bins for numeric features.
    laplace_alpha:
        Laplace smoothing constant for zero-cell protection.

    Returns
    -------
    tuple[pd.DataFrame, float]
        (bin_table with columns [bin, count, n_bad, n_good, pct_bad, pct_good, woe, iv],
         total IV)
    """
    y_arr = np.asarray(y, dtype=float)
    n_bad_total = y_arr.sum()
    n_good_total = len(y_arr) - n_bad_total

    # Bin the feature
    if x.dtype in (object, "category"):
        bins = x.fillna("Missing").astype(str)
    else:
        x_num = pd.to_numeric(x, errors="coerce")
        bins = pd.qcut(x_num, q=n_bins, duplicates="drop")

    records = []
    for b in bins.unique():
        mask = bins == b
        n = int(mask.sum())
        n_b = int(y_arr[mask].sum())
        n_g = n - n_b
        pct_b = (n_b + laplace_alpha) / (n_bad_total + 2 * laplace_alpha)
        pct_g = (n_g + laplace_alpha) / (n_good_total + 2 * laplace_alpha)
        woe = float(np.log(pct_g / pct_b))
        iv = float((pct_g - pct_b) * woe)
        records.append({"bin": str(b), "count": n, "n_bad": n_b, "n_good": n_g,
                         "pct_bad": pct_b, "pct_good": pct_g, "woe": woe, "iv": iv})

    tbl = pd.DataFrame(records)
    total_iv = float(tbl["iv"].sum())
    return tbl, total_iv
