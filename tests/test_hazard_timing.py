"""Regressions for the degenerate-12-month-PD chain.

The discrete hazard model used to give every loan its full contractual term of
person-periods with the default event pinned to the final one and no censoring. The fitted
hazard for the first twelve months was then ~0, so:

  * ``pd_12m`` came back at ~1e-20 for every loan;
  * IFRS 9 Stage 1 ECL was exactly $0.00 across 483,685 loans and $1.24bn of exposure; and
  * ``run_ifrs9_ecl`` overwrote the scorecard's ``pd_12m`` with that figure, so the Phase 9
    cutoff sweep charged an expected loss of zero at every cutoff on the grid.

Nothing in the suite or the QA layer caught any of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.models.pd_term_structure import DiscreteHazardModel
from credit_risk.risk.ifrs9_ecl import IFRS9Config, ScenarioConfig, run_ifrs9_ecl


def _panel(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Loan frame carrying the payment history the duration proxy needs.

    Defaults are placed mid-term (months 6..term-6) rather than at maturity, which is the
    situation the old construction could not represent.
    """
    rng = np.random.default_rng(seed)
    term = rng.choice([36, 60], size=n, p=[0.7, 0.3]).astype(float)
    int_rate = np.clip(rng.normal(0.13, 0.04, n), 0.05, 0.30)
    funded = rng.integers(2_000, 35_000, n).astype(float)
    grade = rng.choice(list("ABCDEFG"), size=n)
    dti = np.clip(rng.normal(18, 7, n), 1, 45)

    is_bad = rng.random(n) < 0.22
    monthly = int_rate / 12.0
    installment = funded * monthly / (1.0 - (1.0 + monthly) ** (-term))

    # Defaulters stop paying somewhere in the middle of the term; survivors run to maturity.
    mob = np.where(is_bad, rng.integers(6, 25, n).astype(float), term)
    mob = np.minimum(mob, term)

    return pd.DataFrame(
        {
            "funded_amnt": funded,
            "loan_amnt": funded,
            "int_rate": int_rate,
            "term": [f" {int(t)} months" for t in term],
            "grade": grade,
            "dti": dti,
            "installment": installment,
            "total_pymnt": installment * mob,
            "delinq_2yrs": rng.integers(0, 3, n).astype(float),
            "target": is_bad.astype(int),
            "issue_d": "Jan-2013",
        }
    )


class TestHazardEventTiming:
    def test_twelve_month_pd_is_not_degenerate(self) -> None:
        """The first twelve months must carry real default probability."""
        df = _panel()
        model = DiscreteHazardModel(max_horizon=60, seed=42).fit(df)
        ts = model.predict_term_structure(df, macro_shock=0.0)

        assert model.duration_basis == "payments_observed_censored"
        assert ts["pd_12m"].mean() > 1e-4, (
            f"mean 12-month PD is {ts['pd_12m'].mean():.3g} — the hazard model is producing "
            "no default probability inside the first year"
        )
        assert (ts["pd_lifetime"] >= ts["pd_12m"] - 1e-9).all()

    def test_duration_basis_falls_back_and_says_so(self) -> None:
        """Without payment history the model must admit it is back on the old panel."""
        df = _panel().drop(columns=["total_pymnt"])
        model = DiscreteHazardModel(max_horizon=60, seed=42).fit(df)
        assert model.duration_basis == "full_term_uncensored"

    def test_censoring_shortens_the_panel(self) -> None:
        """Survivors censored at their observed duration, not carried to full term."""
        df = _panel()
        model = DiscreteHazardModel(max_horizon=60, seed=42)
        terms = np.array([36.0 if "36" in t else 60.0 for t in df["term"]])
        windows = model._observation_windows(df, terms.astype(int))

        assert (windows <= np.minimum(terms, 60)).all()
        assert (
            windows[df["target"].to_numpy() == 1].mean() < terms.mean()
        ), "defaulters should be observed for less than a full term"


class TestPD12mIsNotShadowed:
    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        model = DiscreteHazardModel(max_horizon=60, seed=42).fit(df)
        cfg = IFRS9Config(scenarios=[ScenarioConfig("baseline", 1.0, 0.0)])
        return run_ifrs9_ecl(
            df,
            model,
            lgd_arr=np.full(len(df), 0.75),
            ead_arr=df["funded_amnt"].to_numpy(dtype=float),
            cfg=cfg,
        )

    def test_existing_pd_12m_column_survives(self) -> None:
        """A pd_12m already on the frame is the scorecard's and must not be replaced."""
        df = _panel()
        df["pd_12m"] = 0.0731  # stand-in for the scorecard's annualised PD
        out = self._run(df)

        assert np.allclose(
            out["pd_12m"].to_numpy(dtype=float), 0.0731
        ), "run_ifrs9_ecl overwrote the scorecard's 12-month PD with the hazard model's"
        assert "pd_12m_hazard" in out.columns
        assert out.attrs["ifrs9_summary"]["pd_12m_source"] == "hazard_model"

    def test_pd_12m_written_when_absent(self) -> None:
        out = self._run(_panel())
        assert "pd_12m" in out.columns
        assert out["pd_12m"].equals(out["pd_12m_hazard"])


class TestStageOneIsProvisioned:
    def test_stage1_ecl_is_positive(self) -> None:
        """A populated Stage 1 with a zero provision means the 12-month leg is dead."""
        df = _panel()
        model = DiscreteHazardModel(max_horizon=60, seed=42).fit(df)
        cfg = IFRS9Config(scenarios=[ScenarioConfig("baseline", 1.0, 0.0)])
        out = run_ifrs9_ecl(
            df,
            model,
            lgd_arr=np.full(len(df), 0.75),
            ead_arr=df["funded_amnt"].to_numpy(dtype=float),
            cfg=cfg,
        )

        stage1 = out[out["stage"] == 1]
        assert len(stage1) > 0, "fixture produced no Stage 1 loans; test is vacuous"
        assert (
            float(stage1["ecl"].sum()) > 0.0
        ), f"Stage 1 ECL is zero across {len(stage1)} performing loans"
