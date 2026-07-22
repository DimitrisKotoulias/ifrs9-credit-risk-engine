"""PD Term Structure — discrete-time hazard model.

Computes marginal monthly PD, 12-month PD, and lifetime PD for each loan.

Approach: logistic regression on months-on-book (MOB), grade, and optional
macro overlay. Hazard at each time step estimated from training default events.

    h(t | x) = sigmoid(α₀ + α₁·MOB_t + x'·β + γ·macro_t)

Survival:
    S(t) = ∏_{s=1}^{t} (1 − h(s))
    MarginalPD(t) = S(t−1) · h(t)
    12mPD = 1 − S(12)
    LifetimePD = 1 − S(T)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
_DEFAULT_TERM = 36
_MAX_HORIZON = 60


class DiscreteHazardModel:
    """Monthly discrete-time hazard model for PD term structure.

    Parameters
    ----------
    max_horizon:
        Maximum months to project (caps at loan term).
    macro_gamma:
        Sensitivity to macro shock (multiplier on hazard scale).
    seed:
        Random seed.
    """

    def __init__(
        self,
        max_horizon: int = _MAX_HORIZON,
        macro_gamma: float = 0.3,
        asset_correlation: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.max_horizon = max_horizon
        self.macro_gamma = macro_gamma
        self.asset_correlation = asset_correlation
        self.seed = seed
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] = []
        self._fitted = False

    # ── Feature prep ──────────────────────────────────────────────────────────

    @staticmethod
    def _grade_num(df: pd.DataFrame) -> pd.Series:
        if "grade_num" in df.columns:
            return df["grade_num"]
        if "grade" in df.columns:
            return df["grade"].map(_GRADE_MAP).fillna(4.0)
        return pd.Series(np.full(len(df), 4.0), index=df.index)

    @staticmethod
    def _term_num(df: pd.DataFrame) -> pd.Series:
        if "term_num" in df.columns:
            return df["term_num"]
        if "term" in df.columns:
            return pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            ).fillna(36.0)
        return pd.Series(np.full(len(df), 36.0), index=df.index)

    def _build_features(self, df: pd.DataFrame, mob: np.ndarray) -> np.ndarray:
        """Build feature matrix for a batch of (loan, mob) pairs."""
        grade = self._grade_num(df).values
        int_rate = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
        dti = pd.to_numeric(df.get("dti", 15.0), errors="coerce").fillna(15.0).values
        term = self._term_num(df).values
        return np.column_stack([
            mob,
            mob ** 2,
            grade,
            int_rate,
            dti,
            term,
        ])

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "DiscreteHazardModel":
        """Fit discrete hazard model on loan-level panel data.

        Creates person-period dataset internally: each loan contributes
        one row per month until default or end-of-observation.

        Parameters
        ----------
        df:
            Loan-level DataFrame with `target` (1=default), `term`, `grade`,
            `int_rate`, `dti`.
        target_col:
            Column name of default indicator.
        """
        logger.info("Building person-period dataset for hazard model...")
        terms = self._term_num(df).values.astype(int)
        targets = df[target_col].fillna(0).astype(int).values

        T_all = np.minimum(terms, self.max_horizon)
        # Avoid empty periods if any term is <= 0
        T_all = np.maximum(T_all, 1)

        rep_indices = np.repeat(np.arange(len(df)), T_all)
        mob = np.concatenate([np.arange(1, t + 1) for t in T_all])

        grade = self._grade_num(df).values
        int_rate = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
        dti = pd.to_numeric(df.get("dti", 15.0), errors="coerce").fillna(15.0).values
        term = self._term_num(df).values

        grade_rep = grade[rep_indices]
        int_rate_rep = int_rate[rep_indices]
        dti_rep = dti[rep_indices]
        term_rep = term[rep_indices]

        X = np.column_stack([
            mob,
            mob ** 2,
            grade_rep,
            int_rate_rep,
            dti_rep,
            term_rep,
        ])

        y = np.zeros(len(rep_indices), dtype=int)
        ends = np.cumsum(T_all) - 1
        y[ends] = targets

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(C=0.5, max_iter=500, random_state=self.seed)
        self._model.fit(X_scaled, y)
        self._fitted = True

        event_rate = y.mean()
        logger.info(
            "Hazard model fitted: %d person-periods, event rate=%.4f%%",
            len(y), event_rate * 100,
        )
        return self

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _hazard_at_t(
        self,
        df: pd.DataFrame,
        t: int,
        macro_shock: float = 0.0,
    ) -> np.ndarray:
        """Hazard probability h(t) for each loan at month t."""
        assert self._model is not None and self._scaler is not None
        X = self._build_features(df, np.full(len(df), t, dtype=float))
        X_scaled = self._scaler.transform(X)
        h = self._model.predict_proba(X_scaled)[:, 1]
        # Vasicek single factor credit cycle macro adjustment (Eq. 15):
        # PD_PiT = Phi((Phi^-1(PD_TTC) - sqrt(rho)*Z) / sqrt(1-rho))
        # Convention: macro_shock IS the systematic factor Z.
        # Z < 0 = adverse shock (recession) -> higher PD; Z > 0 = favourable.
        if macro_shock != 0.0:
            from scipy.special import ndtr, ndtri
            h_clipped = np.clip(h, 1e-9, 1 - 1e-9)
            z_ttc = ndtri(h_clipped)
            rho = self.asset_correlation
            z_pit = (z_ttc - np.sqrt(rho) * macro_shock) / np.sqrt(1.0 - rho)
            h = ndtr(z_pit)
        return np.clip(h, 0.0, 1.0)

    def predict_term_structure(
        self,
        df: pd.DataFrame,
        macro_shock: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Compute full PD term structure for each loan.

        Returns
        -------
        dict with keys:
            marginal_pd: ndarray shape (n_loans, max_horizon)
            survival: ndarray shape (n_loans, max_horizon)
            pd_12m: ndarray shape (n_loans,)
            pd_lifetime: ndarray shape (n_loans,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n = len(df)
        terms = self._term_num(df).values.astype(int)
        T = min(int(terms.max()), self.max_horizon)

        # Pre-extract features to avoid pandas overhead in the loop
        grade = self._grade_num(df).values
        int_rate = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
        dti = pd.to_numeric(df.get("dti", 15.0), errors="coerce").fillna(15.0).values
        term = self._term_num(df).values

        survival = np.ones((n, T))
        marginal_pd = np.zeros((n, T))

        for t in range(1, T + 1):
            X = np.column_stack([
                np.full(n, t, dtype=float),
                np.full(n, t ** 2, dtype=float),
                grade,
                int_rate,
                dti,
                term,
            ])
            X_scaled = self._scaler.transform(X)
            h_t = self._model.predict_proba(X_scaled)[:, 1]
            if macro_shock != 0.0:
                from scipy.special import ndtr, ndtri  # noqa: PLC0415
                h_clipped = np.clip(h_t, 1e-9, 1 - 1e-9)
                z_ttc = ndtri(h_clipped)
                rho = self.asset_correlation
                # Eq. 15 convention: macro_shock IS Z; Z < 0 = recession = higher PD
                z_pit = (z_ttc - np.sqrt(rho) * macro_shock) / np.sqrt(1.0 - rho)
                h_t = ndtr(z_pit)
            h_t = np.clip(h_t, 0.0, 1.0)

            s_prev = survival[:, t - 2] if t > 1 else np.ones(n)
            survival[:, t - 1] = s_prev * (1.0 - h_t)
            marginal_pd[:, t - 1] = s_prev * h_t

        pd_12m = 1.0 - (survival[:, 11] if T >= 12 else survival[:, -1])
        # Lifetime PD: 1 - S(term) for each loan's own term
        idx = np.minimum(terms, T) - 1
        pd_lifetime = 1.0 - survival[np.arange(n), idx]

        return {
            "marginal_pd": marginal_pd,
            "survival": survival,
            "pd_12m": pd_12m,
            "pd_lifetime": pd_lifetime,
        }

    def predict_pd_12m(self, df: pd.DataFrame, macro_shock: float = 0.0) -> pd.Series:
        ts = self.predict_term_structure(df, macro_shock)
        return pd.Series(ts["pd_12m"], index=df.index, name="pd_12m")

    def predict_pd_lifetime(self, df: pd.DataFrame, macro_shock: float = 0.0) -> pd.Series:
        ts = self.predict_term_structure(df, macro_shock)
        return pd.Series(ts["pd_lifetime"], index=df.index, name="pd_lifetime")
