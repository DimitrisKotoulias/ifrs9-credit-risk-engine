"""PD Term Structure — discrete-time hazard model.

Computes marginal monthly PD, 12-month PD, and lifetime PD for each loan.

Approach: logistic regression on months-on-book (MOB), grade, and an optional
macro overlay. Hazard at each time step estimated from training default events.

    h(t | x) = sigmoid(α₀ + α₁·MOB_t + α₂·MOB_t² + x'·β)

The macro overlay is applied *after* the logistic fit, as a Vasicek single-factor
transform of the fitted hazard (not as an extra regressor and not as an exp(γ·Z)
scale factor):

    h_PiT(t) = Φ( (Φ⁻¹(h(t)) − √ρ·Z) / √(1−ρ) )

where Z is the systematic factor (Z < 0 = adverse). See `_apply_macro_shock`.

Known limitation: the person-period dataset places every default event in the loan's
final month (no observed default date exists in this data) and applies no censoring,
so the estimated MOB slope is partly an artefact of that construction.
See docs/AUDIT.md finding C4.

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

from credit_risk.risk.ifrs9_ecl import normalize_int_rate_to_fraction

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
    asset_correlation:
        Vasicek asset correlation ρ used by the macro overlay.
    seed:
        Random seed.
    """

    def __init__(
        self,
        max_horizon: int = _MAX_HORIZON,
        asset_correlation: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.max_horizon = max_horizon
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
        # int_rate arrives as a FRACTION (loader.py divides by 100). Filling missing
        # values with the literal 12.0 therefore injected ~92x the column mean straight
        # into StandardScaler; normalising first makes the fallback 0.12 as intended
        # (Flaws.md finding N25).
        int_rate = normalize_int_rate_to_fraction(
            pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
        )
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
        # int_rate arrives as a FRACTION (loader.py divides by 100). Filling missing
        # values with the literal 12.0 therefore injected ~92x the column mean straight
        # into StandardScaler; normalising first makes the fallback 0.12 as intended
        # (Flaws.md finding N25).
        int_rate = normalize_int_rate_to_fraction(
            pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
        )
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

    def _apply_macro_shock(self, h: np.ndarray, macro_shock: float) -> np.ndarray:
        """Vasicek single-factor macro overlay on a fitted hazard.

            h_PiT = Φ( (Φ⁻¹(h) − √ρ·Z) / √(1−ρ) )

        Convention: ``macro_shock`` IS the systematic factor Z.
        Z < 0 = adverse shock (recession) -> higher PD; Z > 0 = favourable.
        """
        if macro_shock == 0.0:
            return np.clip(h, 0.0, 1.0)
        from scipy.special import ndtr, ndtri  # noqa: PLC0415

        h_clipped = np.clip(h, 1e-9, 1 - 1e-9)
        z_ttc = ndtri(h_clipped)
        rho = self.asset_correlation
        z_pit = (z_ttc - np.sqrt(rho) * macro_shock) / np.sqrt(1.0 - rho)
        return np.clip(ndtr(z_pit), 0.0, 1.0)

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
        # int_rate arrives as a FRACTION (loader.py divides by 100). Filling missing
        # values with the literal 12.0 therefore injected ~92x the column mean straight
        # into StandardScaler; normalising first makes the fallback 0.12 as intended
        # (Flaws.md finding N25).
        int_rate = normalize_int_rate_to_fraction(
            pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0).values
        )
        dti = pd.to_numeric(df.get("dti", 15.0), errors="coerce").fillna(15.0).values
        term = self._term_num(df).values

        def _raw_hazard(t: int) -> np.ndarray:
            X = np.column_stack([
                np.full(n, t, dtype=float),
                np.full(n, t ** 2, dtype=float),
                grade,
                int_rate,
                dti,
                term,
            ])
            return self._model.predict_proba(self._scaler.transform(X))[:, 1]

        # ── Macro shock: calibrated annually, so applied annually ──────────────
        #
        # Z is derived by inverting Vasicek on an ANNUAL default rate
        # (risk/ifrs9_ecl.py). Applying that same transform to each MONTHLY hazard
        # compounded it twelve times over: at h ~= 0.005 and Z = -1.64 the monthly hazard
        # rose 3.5x, lifting the cumulative lifetime PD far past the 2.15x ratio the
        # scenario actually targets, and that over-shock fed straight into the
        # probability-weighted ECL (Flaws.md finding N15).
        #
        # Instead the shock is applied to the 12-month cumulative default probability,
        # and the implied uplift is redistributed across months as a proportional-hazards
        # scaling: with S = prod(1 - h_t) over the first 12 months and H = -ln(S),
        #     PD_12m_shocked = Phi((Phi^-1(1 - S) - sqrt(rho) Z) / sqrt(1 - rho))
        #     alpha          = -ln(1 - PD_12m_shocked) / H
        #     h'_t           = 1 - (1 - h_t)^alpha
        # which reproduces the target 12-month PD exactly, since S^alpha = e^{-alpha H}.
        alpha_scale = None
        if macro_shock != 0.0:
            horizon_12 = min(12, T)
            log_surv_12 = np.zeros(n)
            for t in range(1, horizon_12 + 1):
                log_surv_12 += np.log(np.clip(1.0 - _raw_hazard(t), 1e-12, 1.0))
            cum_hazard_12 = -log_surv_12
            pd_12_base = np.clip(1.0 - np.exp(log_surv_12), 1e-9, 1 - 1e-9)
            pd_12_shocked = np.clip(
                self._apply_macro_shock(pd_12_base, macro_shock), 1e-9, 1 - 1e-9
            )
            cum_hazard_shocked = -np.log(1.0 - pd_12_shocked)
            alpha_scale = np.where(
                cum_hazard_12 > 1e-12, cum_hazard_shocked / np.maximum(cum_hazard_12, 1e-12), 1.0
            )

        survival = np.ones((n, T))
        marginal_pd = np.zeros((n, T))

        for t in range(1, T + 1):
            h_t = _raw_hazard(t)
            if alpha_scale is not None:
                h_t = 1.0 - np.power(np.clip(1.0 - h_t, 1e-12, 1.0), alpha_scale)
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
