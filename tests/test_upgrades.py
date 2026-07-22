"""Tests for the integrity / upgrade work:

* single-source-of-truth benchmark registry + QA guards (reports/benchmarks.py,
  reports/qa_checks.py) — the fix for the fabricated LGD R^2 row;
* configurable SICR absolute threshold;
* risk-appetite cutoff approval-rate floor (no degenerate ~0% corner);
* LGD champion-vs-challenger OOS selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# reports/ holds benchmarks.py + qa_checks.py; render_latex adds it to sys.path at build.
_REPORTS = Path(__file__).resolve().parent.parent / "reports"
sys.path.insert(0, str(_REPORTS))

import benchmarks as B  # noqa: E402
import qa_checks as Q  # noqa: E402

from credit_risk.business.cutoff import risk_appetite_cutoff  # noqa: E402
from credit_risk.models.lgd import compute_realised_lgd  # noqa: E402
from credit_risk.utils.config import IFRS9Config, MacroScenario  # noqa: E402
from credit_risk.validation.lgd_validation import validate_lgd_models  # noqa: E402


# ── A. Benchmark registry + QA guards ─────────────────────────────────────────────
class TestBenchmarkRegistry:
    def test_every_benchmark_is_sourced(self):
        for key, b in B.BENCHMARKS.items():
            assert b.source_bibkey, f"{key} missing bibkey"
            assert b.source_locator, f"{key} missing locator"
            assert b.metric_key, f"{key} missing metric_key"
            assert b.low <= b.high, f"{key} inverted range"

    def test_lgd_r2_is_a_live_row_not_a_constant(self):
        # The old bug hard-coded LGD R^2 as a static "0.09-0.15 / Within range".
        lgd = B.BENCHMARKS["LGD_R2"]
        assert lgd.metric_key == "lgd_r2"
        assert lgd.source_bibkey == "loterman2012benchmarking"
        # A negative OOS R^2 must render "Below typical range", not a fabricated pass.
        verdict, _ = lgd.verdict(-1.1333)
        assert verdict == "Below typical range"

    def test_verdict_boundaries(self):
        b = B.BENCHMARKS["MEAN_LGD"]  # 0.25-0.45
        assert b.verdict(0.30)[0] == "Within range"
        assert b.verdict(0.10)[0] == "Below typical range"
        assert b.verdict(0.90)[0] == "Above typical range"
        assert b.verdict(float("nan"))[0] == "N/A"
        assert b.verdict(None)[0] == "N/A"

    def test_range_tex_units(self):
        assert B.BENCHMARKS["MEAN_LGD"].range_tex() == "$0.25 - 0.45$"
        assert "\\%" in B.BENCHMARKS["RWA_DENSITY"].range_tex()

    def test_gini_band_is_reconciled_single_value(self):
        # Table 13 and Table 18 both reference the same GINI_OOT benchmark now.
        assert "GINI_OOT" in B.TABLE13_KEYS
        assert "GINI_OOT" in B.TABLE18_KEYS


class TestQAGuards:
    def test_benchmarks_sourced_passes(self):
        failures: list[str] = []
        Q.check_benchmarks_sourced(failures)
        assert failures == []

    def test_fabricated_benchmark_is_caught(self, tmp_path):
        bad = tmp_path / "r.tex"
        bad.write_text(r"LGD $R^2$ & $0.09 - 0.15$ & within \\", encoding="utf-8")
        failures: list[str] = []
        Q.check_no_fabricated_benchmark(bad, failures)
        assert failures and "0.09" in failures[0]

    def test_clean_tex_passes_fabrication_check(self, tmp_path):
        ok = tmp_path / "r.tex"
        ok.write_text(r"LGD $R^2$ & \textbf{-1.1333} & $0.04 - 0.43$ \\", encoding="utf-8")
        failures: list[str] = []
        Q.check_no_fabricated_benchmark(ok, failures)
        assert failures == []


# ── B. Configurable SICR absolute threshold ───────────────────────────────────────
def _ifrs9(**kw) -> IFRS9Config:
    return IFRS9Config(scenarios={"baseline": MacroScenario(weight=1.0, macro_shock=0.0)}, **kw)


class TestSICRThreshold:
    def test_default_is_020(self):
        assert _ifrs9().sicr.abs_threshold == 0.20

    def test_override_flows_to_sicr(self):
        assert _ifrs9(sicr_abs_threshold=0.35).sicr.abs_threshold == 0.35


# ── C. Cutoff approval-rate floor ─────────────────────────────────────────────────
class TestCutoffFloor:
    def _strategy(self):
        # Bad rate rises as cutoff falls; only the tightest cutoff meets a 0.05 ceiling
        # and only at 0.5% approval — a vacuous corner without a floor.
        return [
            {"cutoff": 700, "approval_rate": 0.005, "bad_rate": 0.04},
            {"cutoff": 600, "approval_rate": 0.40, "bad_rate": 0.12},
            {"cutoff": 500, "approval_rate": 0.80, "bad_rate": 0.20},
        ]

    def test_default_behaviour_unchanged(self):
        # No floor: most inclusive row within the ceiling (only the 0.5% corner qualifies).
        r = risk_appetite_cutoff(self._strategy(), max_bad_rate=0.05)
        assert r["cutoff"] == 700

    def test_floor_avoids_degenerate_corner(self):
        # Require >=20% approval: ceiling unmet at that volume -> best-risk row meeting floor.
        r = risk_appetite_cutoff(self._strategy(), max_bad_rate=0.05, min_approval_rate=0.20)
        assert r["approval_rate"] >= 0.20
        assert r["cutoff"] == 600  # lowest bad rate among rows clearing the floor

    def test_no_row_meets_floor_returns_none(self):
        r = risk_appetite_cutoff(self._strategy(), max_bad_rate=0.05, min_approval_rate=0.99)
        assert r is None


# ── D. LGD champion vs challenger OOS selection ────────────────────────────────────
def _defaults(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    funded = rng.uniform(5_000, 30_000, n)
    total_pymnt = funded * rng.uniform(0.0, 1.0, n)
    total_rec_prncp = total_pymnt * rng.uniform(0.5, 0.95, n)
    return pd.DataFrame({
        "funded_amnt": funded, "total_pymnt": total_pymnt, "total_rec_prncp": total_rec_prncp,
    })


class _MockLGDModel:
    """Minimal LGDModel surface consumed by validate_lgd_models."""

    def __init__(self, champion_const: float) -> None:
        self._use_challenger = False
        self._challenger = object()  # non-None -> challenger is evaluated
        self._champion_const = champion_const

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._champion_const, index=df.index, name="lgd_pred")

    def predict_challenger(self, df: pd.DataFrame) -> pd.Series:
        return compute_realised_lgd(df).rename("lgd_pred")  # perfect -> lower RMSE


class TestLGDSelection:
    def test_recommends_challenger_when_it_wins_oos(self):
        model = _MockLGDModel(champion_const=0.5)
        out = validate_lgd_models(model, _defaults())
        assert out["recommended"] == "challenger"
        assert out["challenger"]["rmse"] < out["champion"]["rmse"]
        assert model._use_challenger is False  # non-mutating: validation only

    def test_champion_kept_when_no_improvement(self):
        # Champion == perfect too: challenger cannot strictly beat it -> keep champion.
        model = _MockLGDModel(champion_const=0.5)
        model.predict = lambda df: compute_realised_lgd(df).rename("lgd_pred")  # type: ignore
        out = validate_lgd_models(model, _defaults(seed=3))
        assert out["recommended"] == "champion"
