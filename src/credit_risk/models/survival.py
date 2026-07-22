"""Survival analysis for the PD term structure (Kaplan-Meier + Cox PH).

The production term structure (``models/pd_term_structure.DiscreteHazardModel``) places
every default at the end of a loan's term, which cannot express *when* within the life a
loan is most likely to default. A proper time-to-event model fixes this: Kaplan-Meier
gives non-parametric survival curves per grade, and a Cox proportional-hazards model
yields covariate hazard ratios plus a time-varying monthly hazard h(t) that is the
industry standard for IFRS 9 lifetime-PD term-structure modelling.

This model is presented **alongside** the production hazard model as a challenger /
robustness analysis; it does not replace the pipeline's ECL driver.

Duration/event are synthesised from the data (no observed default month exists): the
event is the binary default target, and the duration is a months-on-book proxy derived
from cumulative payments (``models/ead.compute_months_on_book_at_default``). This
construction is a documented limitation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from credit_risk.models.ead import compute_months_on_book_at_default
from credit_risk.models.pd_term_structure import _GRADE_MAP

logger = logging.getLogger(__name__)

_COX_COVARIATES = ["grade_num", "int_rate", "dti", "term_num"]


def _grade_series(df: pd.DataFrame) -> pd.Series:
    if "grade_num" in df.columns:
        return pd.to_numeric(df["grade_num"], errors="coerce").fillna(4.0)
    if "grade" in df.columns:
        return df["grade"].map(_GRADE_MAP).fillna(4.0)
    return pd.Series(np.full(len(df), 4.0), index=df.index)


def _term_series(df: pd.DataFrame) -> pd.Series:
    if "term_num" in df.columns:
        return pd.to_numeric(df["term_num"], errors="coerce").fillna(36.0)
    if "term" in df.columns:
        return pd.to_numeric(
            df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        ).fillna(36.0)
    return pd.Series(np.full(len(df), 36.0), index=df.index)


def build_survival_frame(
    df: pd.DataFrame,
    *,
    max_horizon: int = 60,
    target_col: str = "target",
) -> pd.DataFrame:
    """Assemble a (duration, event, grade, covariates) survival dataset.

    ``duration`` is a months-on-book proxy clipped to ``[1, max_horizon]``; ``event`` is
    the binary default target. Returns only the columns needed for KM / Cox fitting.
    """
    out = pd.DataFrame(index=df.index)
    mob = compute_months_on_book_at_default(df)
    out["duration"] = pd.to_numeric(mob, errors="coerce").fillna(
        float(max_horizon) * 0.4
    ).clip(lower=1.0, upper=float(max_horizon))
    out["event"] = (
        pd.to_numeric(df.get(target_col, 0), errors="coerce").fillna(0).clip(0, 1).astype(int)
    )
    out["grade"] = df["grade"].astype(str) if "grade" in df.columns else "NA"
    out["grade_num"] = _grade_series(df).to_numpy()
    out["int_rate"] = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0)
    out["dti"] = pd.to_numeric(df.get("dti", 15.0), errors="coerce").fillna(15.0)
    out["term_num"] = _term_series(df).to_numpy()
    return out.dropna(subset=["duration", "event"])


class SurvivalPDModel:
    """Kaplan-Meier (per grade) + Cox proportional-hazards term-structure model."""

    def __init__(
        self,
        *,
        max_horizon: int = 60,
        penalizer: float = 0.1,
        sample_size: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.max_horizon = max_horizon
        self.penalizer = penalizer
        self.sample_size = sample_size
        self.seed = seed
        self._cph: object | None = None
        self._km_curves: dict[str, pd.DataFrame] = {}
        self._c_index: float = float("nan")
        self._covariates: list[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "target") -> SurvivalPDModel:
        """Fit KM curves per grade and a Cox PH model on a sample of the cohort."""
        from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: PLC0415
        from lifelines.utils import concordance_index  # noqa: PLC0415

        surv = build_survival_frame(df, max_horizon=self.max_horizon, target_col=target_col)
        if len(surv) > self.sample_size:
            surv = surv.sample(self.sample_size, random_state=self.seed)

        # Kaplan-Meier per grade
        kmf = KaplanMeierFitter()
        for grade in sorted(surv["grade"].unique()):
            mask = surv["grade"] == grade
            if int(mask.sum()) < 20:
                continue
            kmf.fit(
                surv.loc[mask, "duration"],
                event_observed=surv.loc[mask, "event"],
                label=str(grade),
            )
            self._km_curves[str(grade)] = kmf.survival_function_.copy()

        # Keep only covariates with variance (Cox fails on constant columns)
        covs = [c for c in _COX_COVARIATES if surv[c].nunique() > 1]
        self._covariates = covs
        cox_data = surv[[*covs, "duration", "event"]].copy()

        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(cox_data, duration_col="duration", event_col="event")
        self._cph = cph

        partial = cph.predict_partial_hazard(surv[covs]).to_numpy().ravel()
        self._c_index = float(
            concordance_index(surv["duration"], -partial, surv["event"])
        )
        self._fitted = True
        logger.info(
            "Survival model fitted: n=%d | grades=%d | Cox C-index=%.4f",
            len(surv), len(self._km_curves), self._c_index,
        )
        return self

    @property
    def concordance(self) -> float:
        return self._c_index

    @property
    def km_curves(self) -> dict[str, pd.DataFrame]:
        return self._km_curves

    def cox_summary(self) -> pd.DataFrame:
        """Return Cox coefficients, hazard ratios and p-values per covariate."""
        if self._cph is None:
            return pd.DataFrame(columns=["covariate", "coef", "hazard_ratio", "p_value"])
        summ = self._cph.summary  # type: ignore[attr-defined]
        return pd.DataFrame({
            "covariate": summ.index.tolist(),
            "coef": summ["coef"].to_numpy(),
            "hazard_ratio": summ["exp(coef)"].to_numpy(),
            "p_value": summ["p"].to_numpy(),
        })

    def monthly_hazard_from_cox(
        self,
        features: pd.DataFrame,
        months: int | None = None,
    ) -> np.ndarray:
        """Extract the monthly conditional hazard h(t) = 1 - S(t)/S(t-1) from Cox.

        Averages the survival function over the supplied ``features`` rows, then
        differences it into monthly hazards — the time-varying replacement for the
        constant-h(t) assumption.
        """
        if self._cph is None:
            raise RuntimeError("SurvivalPDModel must be fitted before hazard extraction")
        months = months or self.max_horizon
        timeline = list(range(1, months + 1))
        # Request the integer timeline explicitly: predict_survival_function's
        # default index is the (fractional) training durations, and testing
        # integer months for membership in that fractional index used to fail
        # for nearly every month, collapsing the hazard curve to ~1 point.
        # Passing `times=` makes lifelines interpolate onto our timeline.
        sf = self._cph.predict_survival_function(  # type: ignore[attr-defined]
            features[self._covariates], times=timeline
        )
        s = sf.loc[timeline].mean(axis=1).to_numpy()
        if len(s) == 0:
            return np.zeros(0)
        hazards = np.zeros(len(s))
        hazards[0] = 1.0 - s[0]
        for t in range(1, len(s)):
            hazards[t] = 1.0 - s[t] / s[t - 1] if s[t - 1] > 0 else 0.0
        return np.clip(hazards, 0.0, 1.0)

    def summary_metrics(self) -> dict[str, float | dict[str, float]]:
        """Compact metrics dict for the pipeline (C-index + per-grade median survival)."""
        median_surv: dict[str, float] = {}
        for grade, sf in self._km_curves.items():
            col = sf.columns[0]
            below = sf.index[sf[col] <= 0.5]
            median_surv[grade] = float(below[0]) if len(below) else float(self.max_horizon)
        return {"c_index": self._c_index, "median_survival_months": median_surv}
