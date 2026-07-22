"""EAD (Exposure at Default) model.

Lending Club loans are fully drawn term instalment loans — no undrawn revolving
facilities. Therefore:

    EAD ≈ outstanding_principal_at_default
        = funded_amnt × amortisation_factor(months_on_book_at_default, term, int_rate)

Amortisation factor via standard annuity formula:
    remaining_principal_fraction(t, T, r) = (1 + r/12)^t × [1 − (1 + r/12)^−T]
                                             / [1 − (1 + r/12)^−(T−t)]
    or equivalently: 1 − (cumulative_principal_paid / funded_amnt)

Credit Conversion Factor (CCF) note:
    CCF is the correct Basel approach for revolving/undrawn exposures (e.g.
    credit cards, lines of credit). For term loans, EAD is deterministic given
    the amortisation schedule. This simplification is documented here because the
    real-world retail IRB model would use CCF for revolving exposures.

Reference: BCBS §318-320.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.risk.ifrs9_ecl import normalize_int_rate_to_fraction

logger = logging.getLogger(__name__)


def _annuity_factor(r: np.ndarray, T: np.ndarray, t: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r_safe = np.where(r < 1e-10, 1e-10, r)
        num = 1.0 - (1.0 + r_safe) ** (-(T - t))
        den = 1.0 - (1.0 + r_safe) ** (-T)
        ratio = np.where(den > 1e-12, num / den, (T - t) / np.where(T > 0, T, 1))
    return np.clip(ratio, 0.0, 1.0)


def amortisation_factor(
    months_on_book: np.ndarray | float,
    term_months: np.ndarray | float,
    annual_rate: np.ndarray | float,
) -> np.ndarray:
    """Fraction of original principal outstanding after 'months_on_book' payments.

    Parameters
    ----------
    months_on_book:
        Number of monthly payments already made (scalar or array).
    term_months:
        Original loan term in months (36 or 60).
    annual_rate:
        Annual interest rate as a fraction (e.g. 0.12 for 12%).

    Returns
    -------
    np.ndarray
        Outstanding principal fraction ∈ [0, 1].
        1.0 at t=0 (no payments made); ≈0 at t=term_months.
    """
    t = np.asarray(months_on_book, dtype=float)
    T = np.asarray(term_months, dtype=float)
    r = np.asarray(annual_rate, dtype=float) / 12.0  # monthly rate
    return _annuity_factor(r, T, t)



def compute_months_on_book_at_default(df: pd.DataFrame) -> pd.Series:
    """Estimate months on book at default for defaulted loans.

    Uses total payments made relative to term length as a proxy when exact
    default date is unavailable.

    Parameters
    ----------
    df:
        DataFrame with 'term' (months), 'total_pymnt' (optional), 'funded_amnt'.

    Returns
    -------
    pd.Series
        Estimated months on book at default.
    """
    term = pd.to_numeric(
        df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(36.0)

    # Proxy: if total_pymnt available, estimate months from payment fraction
    if "total_pymnt" in df.columns:
        total_paid = pd.to_numeric(df["total_pymnt"], errors="coerce").fillna(0.0)
        funded = pd.to_numeric(df["funded_amnt"], errors="coerce").fillna(1.0)
        # int_rate arrives as a PERCENT on Lending Club data (e.g. 12.34), not a
        # fraction. Feeding it raw into the annuity formula makes the monthly rate
        # int_rate/12 ~= 1.0, which explodes the installment and collapses the
        # months-on-book proxy to ~1 for every loan (breaking the KM curves).
        # Normalise to a fraction first (values > 1 are assumed to be percents).
        int_rate = pd.to_numeric(df["int_rate"], errors="coerce").fillna(12.0)
        int_rate = int_rate.where(int_rate <= 1.0, int_rate / 100.0)
        monthly_rate = (int_rate / 12).clip(lower=1e-6)
        installment = (
            funded * monthly_rate / (1 - (1 + monthly_rate) ** (-term))
        ).clip(lower=1.0)
        mob = (total_paid / installment).clip(0.0, term)
    else:
        # Fallback: assume default at 40% of term
        mob = term * 0.4

    return mob.rename("months_on_book")


class EADModel:
    """EAD model for fully-drawn term instalment loans.

    Computes EAD = funded_amnt × amortisation_factor(mob, term, int_rate).
    An optional regression refinement is fitted if empirical R² exceeds a threshold.

    Parameters
    ----------
    min_r2_for_regression:
        Minimum R² (on defaulted loans with observed MOB) to apply regression refinement.
    """

    def __init__(self, min_r2_for_regression: float = 0.20) -> None:
        self.min_r2_for_regression = min_r2_for_regression
        self._use_regression: bool = False
        self._regression: object | None = None
        self._feature_cols: list[str] = []
        self._mean_ead: float = 0.0

    def fit(self, df: pd.DataFrame) -> "EADModel":
        """Fit EAD model on a DataFrame (defaulted or full portfolio).

        Parameters
        ----------
        df:
            DataFrame with 'funded_amnt', 'int_rate', 'term'.
        """
        from sklearn.linear_model import Ridge  # noqa: PLC0415

        mob = compute_months_on_book_at_default(df)
        term = pd.to_numeric(
            df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        ).fillna(36.0)
        # int_rate arrives as a PERCENT on Lending Club data (e.g. 12.34); amortisation_factor
        # expects a fraction (it divides by 12 internally). See compute_months_on_book_at_default.
        int_rate = pd.Series(
            normalize_int_rate_to_fraction(
                pd.to_numeric(df["int_rate"], errors="coerce").fillna(12.0).values
            ),
            index=df.index,
        )
        funded = pd.to_numeric(df["funded_amnt"], errors="coerce").fillna(1.0)

        ead_formula = funded * amortisation_factor(mob.values, term.values, int_rate.values)

        # Optional refinement: regress observed EAD residuals
        if "total_pymnt" in df.columns:
            total_paid = pd.to_numeric(df["total_pymnt"], errors="coerce").fillna(0.0)
            ead_obs = (funded - total_paid.clip(upper=funded)).clip(lower=0.0)
            features = pd.DataFrame({"mob": mob, "term": term, "int_rate": int_rate, "funded": funded})
            reg = Ridge(alpha=1.0)
            reg.fit(features, ead_obs)
            r2 = float(reg.score(features, ead_obs))
            logger.info("EAD regression R²=%.4f (threshold=%.4f)", r2, self.min_r2_for_regression)
            if r2 >= self.min_r2_for_regression:
                self._use_regression = True
                self._regression = reg
                self._feature_cols = list(features.columns)
                logger.info("EAD regression refinement enabled (R²=%.4f).", r2)

        self._mean_ead = float(ead_formula.mean())
        logger.info("EAD model fitted. Mean EAD=%.2f", self._mean_ead)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict EAD for each loan."""
        mob = compute_months_on_book_at_default(df)
        term = pd.to_numeric(
            df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        ).fillna(36.0)
        int_rate = pd.Series(
            normalize_int_rate_to_fraction(
                pd.to_numeric(df["int_rate"], errors="coerce").fillna(12.0).values
            ),
            index=df.index,
        )
        funded = pd.to_numeric(df["funded_amnt"], errors="coerce").fillna(1.0)

        ead = funded * amortisation_factor(mob.values, term.values, int_rate.values)

        if self._use_regression and self._regression is not None:
            features = pd.DataFrame({
                "mob": mob, "term": term, "int_rate": int_rate, "funded": funded
            }, index=df.index)
            reg_pred = self._regression.predict(features)
            ead = pd.Series(np.clip(reg_pred, 0.0, funded.values), index=df.index)

        return pd.Series(ead, index=df.index, name="ead").clip(lower=0.0)

    @property
    def mean_ead(self) -> float:
        return self._mean_ead

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "EADModel":
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301
