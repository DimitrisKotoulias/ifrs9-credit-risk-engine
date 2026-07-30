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

Event timing and censoring: this data records no observed default *date*, so the
person-period dataset is built against a duration proxy — cumulative payments divided by
the contractual instalment (``models/ead.compute_months_on_book_at_default``), the same
proxy the Kaplan-Meier/Cox challenger in ``models/survival.py`` uses. Defaulters contribute
periods 1..d with the event at d; survivors contribute periods 1..d_obs, right-censored.

This replaces the original construction, which gave every loan its FULL contractual term of
person-periods and pinned every default event to the final one, with no censoring. That
made the fitted hazard ~0 for the first twelve months of every loan, so ``pd_12m`` came back
at ~1e-20 and the IFRS 9 Stage 1 (12-month ECL) leg provisioned exactly zero across half
the book (AUDIT-C4, whose consequence was disclosed as an artefactual MOB
slope but never quantified).

Residual limitation: the duration is still a proxy, not an observed default month, and
prepayment is treated as censoring rather than as a competing risk.

Survival:
    S(t) = ∏_{s=1}^{t} (1 − h(s))
    MarginalPD(t) = S(t−1) · h(t)
    12mPD = 1 − S(12)
    LifetimePD = 1 − S(T)
"""

from __future__ import annotations

import logging

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
        # Which duration the person-period panel was built against. Surfaced into
        # metrics.json by the pipeline so a silent fallback to the uncensored full-term
        # construction cannot pass unnoticed.
        self._duration_basis: str = "unknown"

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
        # (FLAWS-N25).
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

    def _observation_windows(self, df: pd.DataFrame, terms: np.ndarray) -> np.ndarray:
        """Per-loan number of person-periods, i.e. the observed duration.

        Defaulters are observed until their (proxied) default month; survivors are
        right-censored at the month their payments stop. Falls back to the full contractual
        term — the original, uncensored construction — when no payment history is available
        to infer a duration from, which is the case for synthetic fixtures.
        """
        from credit_risk.models.ead import (
            compute_months_on_book_at_default,  # noqa: PLC0415
            months_on_book_basis,  # noqa: PLC0415
        )

        cap = np.maximum(np.minimum(terms, self.max_horizon), 1)

        # `months_on_book_basis(df, None)` skips the reporting-date branch: here we want
        # each loan's own observed duration, not its age at a portfolio reporting date.
        basis = months_on_book_basis(df, None)
        if basis != "payments_observed":
            self._duration_basis = "full_term_uncensored"
            logger.warning(
                "Hazard model: no payment history to infer a default month from, so the "
                "person-period panel falls back to the FULL contractual term with every "
                "event pinned to the final period and no censoring. The fitted 12-month "
                "hazard will be ~0 and IFRS 9 Stage 1 ECL will be ~0 (the internal audit log C4)."
            )
            return cap

        mob = pd.to_numeric(
            compute_months_on_book_at_default(df), errors="coerce"
        ).to_numpy(dtype=float)
        # Round up: a loan observed for 5.4 months has been through 6 monthly periods.
        window = np.ceil(np.nan_to_num(mob, nan=0.0))
        window = np.clip(window, 1.0, cap).astype(int)
        self._duration_basis = "payments_observed_censored"
        logger.info(
            "Hazard model duration basis: %s (mean observed window=%.1f months vs mean "
            "contractual term=%.1f)",
            self._duration_basis, float(window.mean()), float(cap.mean()),
        )
        return window

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> "DiscreteHazardModel":
        """Fit discrete hazard model on loan-level panel data.

        Creates a person-period dataset internally: each loan contributes one row per month
        until it defaults or its observation window ends (right-censoring). See the module
        docstring for how the duration is proxied.

        Parameters
        ----------
        df:
            Loan-level DataFrame with `target` (1=default), `term`, `grade`, `int_rate`,
            `dti`. `total_pymnt`/`funded_amnt` are used, when present, ONLY to place the
            event in time — never as predictors — so the leakage policy is unaffected.
        target_col:
            Column name of default indicator.
        """
        logger.info("Building person-period dataset for hazard model...")
        terms = self._term_num(df).values.astype(int)
        targets = df[target_col].fillna(0).astype(int).values

        T_all = self._observation_windows(df, terms)

        rep_indices = np.repeat(np.arange(len(df)), T_all)
        mob = np.concatenate([np.arange(1, t + 1) for t in T_all])

        grade = self._grade_num(df).values
        # int_rate arrives as a FRACTION (loader.py divides by 100). Filling missing
        # values with the literal 12.0 therefore injected ~92x the column mean straight
        # into StandardScaler; normalising first makes the fallback 0.12 as intended
        # (FLAWS-N25).
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
            "Hazard model fitted: %d person-periods, event rate=%.4f%% (duration basis: %s)",
            len(y), event_rate * 100, self._duration_basis,
        )
        return self

    @property
    def duration_basis(self) -> str:
        """How the person-period observation windows were derived — see `fit`."""
        return self._duration_basis

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
        # (FLAWS-N25).
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

        # ── Macro shock: applied once, at the horizon Z is calibrated on ───────
        #
        # Z is derived by inverting Vasicek on a portfolio default rate
        # (risk/ifrs9_ecl.py, `ttc_dr = mean(target)`). Because the target is the loan's
        # terminal resolved status, that rate is a LIFETIME default rate. Applying the
        # same transform to each MONTHLY hazard compounded it over the whole term, badly
        # over-shocking the downside scenario and inflating the probability-weighted ECL
        # (FLAWS-N15).
        #
        # The shock is therefore applied ONCE, to each loan's cumulative default
        # probability over its own term, and the resulting uplift is redistributed across
        # months as a proportional-hazards scaling: with S = prod(1 - h_t) to the end of
        # the term and H = -ln(S),
        #     PD_shocked = Phi((Phi^-1(1 - S) - sqrt(rho) Z) / sqrt(1 - rho))
        #     alpha      = -ln(1 - PD_shocked) / H
        #     h'_t       = 1 - (1 - h_t)^alpha
        # which reproduces the target lifetime PD exactly, since S^alpha = e^{-alpha H}.
        #
        # The anchor must be the lifetime horizon because that is the horizon Z is
        # calibrated on: risk/ifrs9_ecl.py inverts Vasicek on `ttc_dr = mean(target)`, and
        # the target is the loan's terminal resolved status, i.e. a LIFETIME default rate.
        # Anchoring the transform at twelve months would apply a lifetime-calibrated Z to a
        # twelve-month probability, which is a different quantity.
        #
        # (Historically there was a second reason: the panel pinned every event to the
        # loan's final month, so the 12-month cumulative hazard was ~0 and a 12-month anchor
        # made alpha identically 1, silently disabling the macro overlay. The panel is now
        # built with real durations and censoring, so that failure mode is gone; the
        # calibration-horizon argument above is the one that still stands.)
        alpha_scale = None
        if macro_shock != 0.0:
            # Cumulative hazard to the end of each loan's own term.
            term_idx = np.minimum(terms, T)
            log_surv_life = np.zeros(n)
            for t in range(1, T + 1):
                active = term_idx >= t
                if not active.any():
                    break
                log_surv_life += np.where(
                    active, np.log(np.clip(1.0 - _raw_hazard(t), 1e-12, 1.0)), 0.0
                )
            cum_hazard_life = -log_surv_life
            pd_life_base = np.clip(1.0 - np.exp(log_surv_life), 1e-9, 1 - 1e-9)
            pd_life_shocked = np.clip(
                self._apply_macro_shock(pd_life_base, macro_shock), 1e-9, 1 - 1e-9
            )
            cum_hazard_shocked = -np.log(1.0 - pd_life_shocked)
            alpha_scale = np.where(
                cum_hazard_life > 1e-12,
                cum_hazard_shocked / np.maximum(cum_hazard_life, 1e-12),
                1.0,
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
