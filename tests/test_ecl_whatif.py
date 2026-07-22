"""Tests for the ECL what-if / shock sensitivity calculator."""

import numpy as np
import pandas as pd

from credit_risk.risk.ifrs9_ecl import (
    DEFAULT_SHOCK_SCENARIOS,
    ecl_shock_sensitivity,
)


class _FakeHazard:
    """Constant marginal-PD term structure, independent of macro shock."""

    def __init__(self, n: int, horizon: int = 24, pd_val: float = 0.01) -> None:
        self.n = n
        self.horizon = horizon
        self.pd_val = pd_val

    def predict_term_structure(self, df: pd.DataFrame, macro_shock: float = 0.0) -> dict:
        mp = np.full((len(df), self.horizon), self.pd_val)
        return {"marginal_pd": mp}


def _setup(n: int = 100):
    df = pd.DataFrame({"int_rate": np.full(n, 10.0)})
    lgd = np.full(n, 0.5)
    ead = np.full(n, 10_000.0)
    stages = np.full(n, 2)  # all lifetime => ECL is linear in marginal PD
    return df, lgd, ead, stages, _FakeHazard(n)


def test_identity_scenario_zero_delta():
    df, lgd, ead, stages, hz = _setup()
    out = ecl_shock_sensitivity(df, hz, lgd, ead, stages, {"identity": {}})
    assert abs(float(out.iloc[0]["delta_ecl"])) < 1e-6
    assert abs(float(out.iloc[0]["delta_pct"])) < 1e-6


def test_pd_doubling_doubles_ecl_for_lifetime():
    df, lgd, ead, stages, hz = _setup()
    out = ecl_shock_sensitivity(df, hz, lgd, ead, stages, {"pd2x": {"pd_multiplier": 2.0}})
    row = out.iloc[0]
    assert row["shocked_ecl"] > row["base_ecl"]
    assert abs(row["shocked_ecl"] / row["base_ecl"] - 2.0) < 1e-6


def test_monotone_in_pd_multiplier():
    df, lgd, ead, stages, hz = _setup()
    out = ecl_shock_sensitivity(
        df, hz, lgd, ead, stages,
        {"a": {"pd_multiplier": 1.2}, "b": {"pd_multiplier": 1.5}, "c": {"pd_multiplier": 2.0}},
    )
    deltas = out.set_index("scenario")["delta_ecl"]
    assert deltas["a"] < deltas["b"] < deltas["c"]


def test_lgd_add_increases_ecl():
    df, lgd, ead, stages, hz = _setup()
    out = ecl_shock_sensitivity(df, hz, lgd, ead, stages, {"lgd": {"lgd_add": 0.1}})
    assert float(out.iloc[0]["delta_ecl"]) > 0.0


def test_default_scenarios_all_present_and_columns():
    df, lgd, ead, stages, hz = _setup()
    out = ecl_shock_sensitivity(df, hz, lgd, ead, stages)
    assert len(out) == len(DEFAULT_SHOCK_SCENARIOS)
    assert {"scenario", "base_ecl", "shocked_ecl", "delta_ecl", "delta_pct"}.issubset(out.columns)
    # Severe scenarios must raise ECL.
    assert float(out.set_index("scenario").loc["GFC-like (PD x3.0)", "delta_ecl"]) > 0.0
