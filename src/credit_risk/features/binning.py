"""WoE/IV monotonic binning.

Primary: optbinning.BinningProcess with monotonic trend enforcement.
Fallback: manual MonotonicBinner using quantile + isotonic regression merge.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def _try_optbinning() -> bool:
    """Return True only if optbinning is importable."""
    try:
        from optbinning import BinningProcess  # noqa: PLC0415
        return True
    except (ImportError, Exception):
        return False


class OptBinningWrapper(BaseEstimator, TransformerMixin):
    """Wrapper over optbinning.BinningProcess for WoE-monotonic binning.

    Parameters
    ----------
    variables:
        Names of features to bin. Remaining columns pass through.
    monotonic_trend:
        Trend constraint per variable; 'auto' lets optbinning decide.
    max_n_bins:
        Maximum number of bins per feature.
    min_bin_frac:
        Minimum fraction of observations per bin.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        monotonic_trend: str = "auto",
        max_n_bins: int = 10,
        min_bin_frac: float = 0.05,
    ) -> None:
        self.variables = variables
        self.monotonic_trend = monotonic_trend
        self.max_n_bins = max_n_bins
        self.min_bin_frac = min_bin_frac
        self._process: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OptBinningWrapper":
        from optbinning import BinningProcess  # noqa: PLC0415

        variables = self.variables or list(X.select_dtypes(include="number").columns)
        self.variables_ = [v for v in variables if v in X.columns]

        # Categorical variables use special handling
        categorical = list(X.select_dtypes(exclude="number").columns)
        categorical_vars = [c for c in categorical if c in (self.variables or [])]

        variable_dtypes = {}
        for v in self.variables_:
            if X[v].dtype == object or X[v].dtype.name == "category":
                variable_dtypes[v] = "categorical"

        self._process = BinningProcess(
            variable_names=self.variables_,
            categorical_variables=list(variable_dtypes.keys()),
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_frac,
        )
        self._process.fit(X[self.variables_], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._process is None:
            raise RuntimeError("Call fit() before transform().")
        woe_arr = self._process.transform(X[self.variables_], metric="woe")
        if hasattr(woe_arr, "values"):
            woe_arr = woe_arr.values
        else:
            woe_arr = np.asarray(woe_arr)
        out = X.copy()
        for i, col in enumerate(self.variables_):
            out[col] = woe_arr[:, i] if woe_arr.ndim > 1 else woe_arr
        return out

    def get_iv_table(self) -> pd.DataFrame:
        """Return IV per variable as a DataFrame."""
        if self._process is None:
            raise RuntimeError("Call fit() first.")
        summary = self._process.summary()
        return summary[["name", "iv"]].rename(columns={"name": "variable"})


class ManualMonotonicBinner(BaseEstimator, TransformerMixin):
    """Fallback monotonic binner using quantile splits + isotonic WoE merge.

    Uses quantile-based initial binning then merges adjacent bins that violate
    monotonicity in WoE until the trend is monotone.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        n_initial_bins: int = 20,
        min_bin_frac: float = 0.05,
        laplace_alpha: float = 0.5,
    ) -> None:
        self.variables = variables
        self.n_initial_bins = n_initial_bins
        self.min_bin_frac = min_bin_frac
        self.laplace_alpha = laplace_alpha

        # Fitted state
        self.bin_edges_: dict[str, list[float]] = {}
        self.woe_maps_: dict[str, dict[int, float]] = {}
        self.iv_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ManualMonotonicBinner":
        y_arr = np.asarray(y, dtype=float)
        n = len(y_arr)
        variables = self.variables or list(X.select_dtypes(include="number").columns)
        self.variables_ = [v for v in variables if v in X.columns]

        for col in self.variables_:
            x_arr = pd.to_numeric(X[col], errors="coerce").values
            edges, woe_map, iv = self._fit_one(x_arr, y_arr, n)
            self.bin_edges_[col] = edges
            self.woe_maps_[col] = woe_map
            self.iv_[col] = iv

        return self

    def _fit_one(
        self, x: np.ndarray, y: np.ndarray, n: int
    ) -> tuple[list[float], dict[int, float], float]:
        """Bin a single numeric feature monotonically."""
        alpha = self.laplace_alpha
        n_total = float(len(y))
        n_bad = float(y.sum())
        n_good = n_total - n_bad

        # Initial quantile bins (ignore NaN)
        valid_mask = ~np.isnan(x)
        q = np.nanpercentile(x[valid_mask], np.linspace(0, 100, self.n_initial_bins + 1))
        edges = sorted(set(q))
        edges[0] = -np.inf
        edges[-1] = np.inf

        def _bins_to_woe(edges: list[float]) -> tuple[list[float], list[float]]:
            bin_ids = np.digitize(x, edges[1:-1], right=True)
            woes = []
            ivs = []
            for b in range(len(edges) - 1):
                mask = (bin_ids == b) & valid_mask
                bg = float((y[mask] == 0).sum())
                bb = float((y[mask] == 1).sum())
                pct_g = (bg + alpha) / (n_good + 2 * alpha)
                pct_b = (bb + alpha) / (n_bad + 2 * alpha)
                woe_val = float(np.log(pct_g / pct_b))
                iv_val = float((pct_g - pct_b) * woe_val)
                woes.append(woe_val)
                ivs.append(iv_val)
            return woes, ivs

        # Merge bins to enforce monotonicity (descending WoE → higher score = lower PD)
        for _ in range(100):
            woes, _ = _bins_to_woe(edges)
            # Check monotone: all diffs same sign
            diffs = np.diff(woes)
            nonzero = diffs[diffs != 0]
            if len(nonzero) == 0:
                break
            directions = np.sign(nonzero)
            if np.all(directions >= 0) or np.all(directions <= 0):
                break
            # Find first violation and merge those bins
            for i in range(len(diffs)):
                if i > 0 and np.sign(diffs[i]) != np.sign(diffs[i - 1]):
                    # Merge bin i and i+1
                    edges = edges[:i + 1] + edges[i + 2:]
                    break
            if len(edges) <= 3:
                break

        woes, ivs = _bins_to_woe(edges)
        woe_map = {b: w for b, w in enumerate(woes)}
        total_iv = float(sum(ivs))

        return edges, woe_map, total_iv

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.variables_:
            edges = self.bin_edges_[col]
            woe_map = self.woe_maps_[col]
            x_arr = pd.to_numeric(X[col], errors="coerce").values
            valid = ~np.isnan(x_arr)
            bin_ids = np.zeros(len(x_arr), dtype=int)
            bin_ids[valid] = np.digitize(x_arr[valid], edges[1:-1], right=True)
            # Missing bin: use bin 0 (first bin's WoE approximation)
            out[col] = pd.Series(bin_ids).map(woe_map).fillna(0.0).values
        return out

    def get_iv_table(self) -> pd.DataFrame:
        if not self.iv_:
            raise RuntimeError("Call fit() first.")
        return pd.DataFrame(
            {"variable": list(self.iv_.keys()), "iv": list(self.iv_.values())}
        )


def get_binner(
    variables: list[str] | None = None,
    max_n_bins: int = 10,
    min_bin_frac: float = 0.05,
    **_extra: Any,
) -> OptBinningWrapper | ManualMonotonicBinner:
    """Return best available binner (optbinning if installed, else manual fallback)."""
    if _try_optbinning():
        logger.info("Using optbinning for WoE/IV binning.")
        return OptBinningWrapper(
            variables=variables, max_n_bins=max_n_bins, min_bin_frac=min_bin_frac
        )
    logger.warning("optbinning not available; using ManualMonotonicBinner fallback.")
    return ManualMonotonicBinner(
        variables=variables, n_initial_bins=max_n_bins * 2, min_bin_frac=min_bin_frac
    )
