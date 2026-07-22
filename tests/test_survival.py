"""Tests for the Kaplan-Meier + Cox survival PD model."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.models.survival import SurvivalPDModel, build_survival_frame

pytest.importorskip("lifelines")


def _synthetic_cohort(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grades = rng.choice(list("ABCDEFG"), n)
    grade_num = np.array([{"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}[g]
                          for g in grades])
    int_rate = 0.05 + grade_num * 0.02 + rng.normal(0, 0.005, n)
    dti = rng.uniform(5, 30, n)
    funded = rng.uniform(5_000, 30_000, n)
    term = rng.choice([36, 60], n)
    # Higher grade number => higher default probability.
    p_default = np.clip(0.02 + grade_num * 0.04, 0.02, 0.5)
    target = (rng.uniform(0, 1, n) < p_default).astype(int)
    # Defaulted loans pay only a fraction; performing loans pay near-full.
    frac = np.where(target == 1, rng.uniform(0.1, 0.5, n), rng.uniform(0.8, 1.0, n))
    total_pymnt = funded * frac
    return pd.DataFrame({
        "grade": grades, "int_rate": int_rate, "dti": dti,
        "funded_amnt": funded, "term": term, "total_pymnt": total_pymnt,
        "target": target,
    })


def test_build_survival_frame_columns_and_bounds():
    df = _synthetic_cohort()
    surv = build_survival_frame(df, max_horizon=60)
    assert {"duration", "event", "grade", "grade_num"}.issubset(surv.columns)
    assert (surv["duration"] >= 1.0).all() and (surv["duration"] <= 60.0).all()
    assert set(surv["event"].unique()).issubset({0, 1})


def test_cox_fit_and_cindex():
    model = SurvivalPDModel(max_horizon=60, seed=1).fit(_synthetic_cohort(seed=1))
    assert 0.0 <= model.concordance <= 1.0
    summ = model.cox_summary()
    assert not summ.empty
    assert {"covariate", "coef", "hazard_ratio", "p_value"}.issubset(summ.columns)


def test_km_curves_monotone_decreasing():
    model = SurvivalPDModel(max_horizon=60, seed=2).fit(_synthetic_cohort(seed=2))
    assert len(model.km_curves) >= 2
    for sf in model.km_curves.values():
        s = sf.iloc[:, 0].to_numpy()
        assert np.all(np.diff(s) <= 1e-9)  # survival is non-increasing
        assert s.max() <= 1.0 + 1e-9 and s.min() >= -1e-9


def test_monthly_hazard_non_negative():
    df = _synthetic_cohort(seed=3)
    model = SurvivalPDModel(max_horizon=36, seed=3).fit(df)
    surv = build_survival_frame(df, max_horizon=36)
    h = model.monthly_hazard_from_cox(surv.head(50), months=36)
    assert np.all(h >= 0.0) and np.all(h <= 1.0)


def test_summary_metrics_shape():
    model = SurvivalPDModel(seed=4).fit(_synthetic_cohort(seed=4))
    m = model.summary_metrics()
    assert "c_index" in m and "median_survival_months" in m
    assert isinstance(m["median_survival_months"], dict)
