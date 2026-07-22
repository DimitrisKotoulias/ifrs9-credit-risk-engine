"""Tests for model validation: PSI, Gini, KS, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


class TestDiscrimination:
    def test_gini_equals_2_auc_minus_1(self, rng: np.random.Generator) -> None:
        from credit_risk.validation.discrimination import compute_discrimination

        y = (rng.random(500) > 0.7).astype(int)
        score = rng.random(500)
        result = compute_discrimination(y, score)
        auc = roc_auc_score(y, score)
        assert abs(result["gini"] - (2 * auc - 1)) < 1e-10

    def test_ks_in_0_1(self, rng: np.random.Generator) -> None:
        from credit_risk.validation.discrimination import compute_discrimination

        y = (rng.random(200) > 0.8).astype(int)
        score = rng.random(200)
        result = compute_discrimination(y, score)
        assert 0.0 <= result["ks"] <= 1.0

    def test_perfect_model_high_auc(self) -> None:
        from credit_risk.validation.discrimination import compute_discrimination

        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        result = compute_discrimination(y, score)
        assert result["auc"] >= 0.99

    def test_random_model_auc_near_05(self, rng: np.random.Generator) -> None:
        from credit_risk.validation.discrimination import compute_discrimination

        y = np.repeat([0, 1], 5000)
        score = rng.random(10000)
        result = compute_discrimination(y, score)
        assert abs(result["auc"] - 0.5) < 0.05


class TestPSI:
    def test_identical_distributions_psi_near_zero(self, rng: np.random.Generator) -> None:
        from credit_risk.validation.stability import compute_psi

        scores = rng.random(1000)
        psi = compute_psi(scores, scores.copy())
        assert psi < 0.01

    def test_different_distributions_psi_positive(self, rng: np.random.Generator) -> None:
        from credit_risk.validation.stability import compute_psi

        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(2, 1, 1000)  # clearly shifted
        psi = compute_psi(expected, actual)
        assert psi > 0.25, f"Expected PSI > 0.25 for shifted distribution, got {psi:.4f}"

    def test_psi_formula_matches_appendix_d(self, rng: np.random.Generator) -> None:
        """Verify PSI formula: shifted distribution gives PSI in expected range."""
        from credit_risk.validation.stability import compute_psi

        # Two continuous distributions 1 std apart — large, measurable shift
        exp = rng.normal(0, 1, 2000)
        act = rng.normal(1.5, 1, 2000)  # shifted by 1.5 stds
        psi = compute_psi(exp, act, n_bins=10)
        # Large shift → PSI should clearly exceed 0.10
        assert psi > 0.10, f"PSI={psi:.4f} — expected > 0.10 for 1.5-std shift"

        # Same distribution → PSI near zero
        exp2 = rng.normal(0, 1, 2000)
        act2 = rng.normal(0, 1, 2000)
        psi_stable = compute_psi(exp2, act2, n_bins=10)
        assert psi_stable < 0.10, f"PSI={psi_stable:.4f} — same distribution should be stable"

    def test_psi_band_labels(self) -> None:
        from credit_risk.validation.stability import psi_band

        assert psi_band(0.05) == "stable"
        assert psi_band(0.15) == "moderate_shift"
        assert psi_band(0.30) == "significant_shift"


class TestCalibration:
    def test_brier_score_perfect_model(self) -> None:
        from credit_risk.validation.calibration import compute_calibration

        y = np.array([0, 0, 1, 1])
        pred = np.array([0.01, 0.01, 0.99, 0.99])
        result = compute_calibration(y, pred, n_bins=4)
        assert result["brier_score"] < 0.01

    def test_brier_score_random_model(self, rng: np.random.Generator) -> None:
        from credit_risk.validation.calibration import compute_calibration

        y = (rng.random(1000) > 0.8).astype(int)
        pred = np.full(1000, y.mean())  # constant prediction = naive
        result = compute_calibration(y, pred, n_bins=10)
        # Brier score for constant model: p*(1-p)
        expected_brier = y.mean() * (1 - y.mean())
        assert abs(result["brier_score"] - expected_brier) < 0.02

    def test_calibration_keys(self, rng: np.random.Generator) -> None:
        from credit_risk.validation.calibration import compute_calibration

        y = (rng.random(500) > 0.7).astype(int)
        pred = np.clip(rng.random(500) * 0.5 + 0.1, 0, 1)
        result = compute_calibration(y, pred)
        assert {"brier_score", "hl_statistic", "hl_pvalue", "n_bins"} == set(result.keys())


class TestNewValidation:
    def test_bootstrap_auc_ci(self) -> None:
        from credit_risk.validation.discrimination import bootstrap_auc_ci

        rng = np.random.default_rng(42)
        y = (rng.random(300) > 0.7).astype(int)
        pred = rng.random(300)
        mean_auc, lower, upper = bootstrap_auc_ci(y, pred, n_boot=200, seed=42)
        assert isinstance(mean_auc, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= mean_auc <= upper

    def test_spiegelhalter_test(self) -> None:
        from credit_risk.validation.calibration import spiegelhalter_test

        rng = np.random.default_rng(0)
        # Roughly calibrated: draw outcomes from predicted probabilities
        pred = rng.uniform(0.05, 0.40, 600)
        y = (rng.random(600) < pred).astype(int)
        result = spiegelhalter_test(y, pred)
        assert "z_stat" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert abs(result["z_stat"]) < 5  # near-calibrated data → small z

    def test_delong_test(self) -> None:
        from credit_risk.validation.discrimination import delong_test

        rng = np.random.default_rng(7)
        y = (rng.random(400) > 0.7).astype(int)
        pred_a = rng.random(400)
        pred_b = np.clip(pred_a + rng.normal(0, 0.05, 400), 0, 1)
        result = delong_test(y, pred_a, pred_b)
        assert 0.0 <= result["p_value"] <= 1.0
        assert "z_stat" in result
        assert "auc_a" in result and "auc_b" in result

    def test_vintage_pd_accuracy(self) -> None:
        from credit_risk.validation.backtest import vintage_pd_accuracy

        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "issue_d": ["2015-01"] * 120 + ["2016-01"] * 120,
            "pd_pred": list(rng.uniform(0.05, 0.30, 120)) + list(rng.uniform(0.05, 0.30, 120)),
            "target": list((rng.random(120) < 0.15).astype(int))
                      + list((rng.random(120) < 0.15).astype(int)),
        })
        result = vintage_pd_accuracy(df)
        assert "pd_ratio" in result.columns
        assert result["pd_ratio"].notna().all()
        assert len(result) >= 2

    def test_compute_csi(self) -> None:
        from credit_risk.validation.stability import compute_csi

        rng = np.random.default_rng(99)
        x_train = pd.DataFrame({
            "feat_a": rng.normal(0, 1, 500),
            "feat_b": rng.uniform(0, 1, 500),
        })
        x_oot = pd.DataFrame({
            "feat_a": rng.normal(0, 1, 200),
            "feat_b": rng.uniform(0, 1, 200),
        })
        result = compute_csi(x_train, x_oot, ["feat_a", "feat_b"])
        assert isinstance(result, pd.DataFrame)
        assert "csi" in result.columns
        assert result["csi"].mean() < 0.10


def test_rag_green():
    from credit_risk.validation.discrimination import RAGStatus
    rag = RAGStatus(gini_train=0.40, gini_oot=0.38, psi=0.05)
    assert rag.gini_rag == "GREEN" and rag.psi_rag == "GREEN" and rag.overall == "GREEN"

def test_rag_amber_gini():
    from credit_risk.validation.discrimination import RAGStatus
    rag = RAGStatus(gini_train=0.40, gini_oot=0.33, psi=0.05)
    assert rag.gini_rag == "AMBER" and rag.overall == "AMBER"

def test_rag_red_psi():
    from credit_risk.validation.discrimination import RAGStatus
    rag = RAGStatus(gini_train=0.40, gini_oot=0.39, psi=0.30)
    assert rag.psi_rag == "RED" and rag.overall == "RED"

def test_hosmer_lemeshow_perfect_calibration():
    import numpy as np
    from credit_risk.validation.calibration import hosmer_lemeshow_test
    rng = np.random.default_rng(42)
    p = rng.uniform(0.1, 0.9, 2000)
    y = rng.binomial(1, p).astype(float)
    result = hosmer_lemeshow_test(y, p)
    assert result["p_value"] > 0.05, f"p={result['p_value']:.4f}"

def test_hosmer_lemeshow_miscalibrated():
    import numpy as np
    from credit_risk.validation.calibration import hosmer_lemeshow_test
    y = np.concatenate([np.ones(500), np.zeros(500)])
    p = np.full(1000, 0.1)
    result = hosmer_lemeshow_test(y, p)
    assert result["p_value"] < 0.05

def test_vintage_pd_calibration_flags():
    import pandas as pd, numpy as np
    from credit_risk.validation.backtest import vintage_pd_accuracy
    df = pd.DataFrame({
        "issue_d": (["2013-01-01"] * 100 + ["2014-01-01"] * 100),
        "pd_pred": np.full(200, 0.10),
        "target": np.concatenate([
            np.concatenate([np.ones(10), np.zeros(90)]),
            np.concatenate([np.ones(50), np.zeros(50)]),
        ]),
    })
    result = vintage_pd_accuracy(df, pd_col="pd_pred", target_col="target", vintage_col="issue_d")
    flags = {str(r["vintage"])[:4]: r["calibration_flag"] for _, r in result.iterrows()}
    assert flags["2013"] == "pass"
    assert flags["2014"] == "fail"


def test_benchmark_convergence() -> None:
    import numpy as np
    from credit_risk.utils.config import load_config
    from credit_risk.data.loader import load_and_prepare
    from credit_risk.models.pd_scorecard import PDScorecard
    from credit_risk.validation.discrimination import compute_discrimination
    from credit_risk.validation.calibration import compute_calibration
    from credit_risk.validation.stability import compute_psi

    seeds = [42, 100, 2026]
    ginis = []
    aucs = []
    briers = []
    psis = []

    for seed in seeds:
        cfg = load_config()
        cfg.data.source = "synthetic"
        cfg.data.synthetic_n_loans = 5000
        cfg.random_seed = seed

        split, _ = load_and_prepare(cfg)
        df_train = split.train
        df_test = split.test
        df_oot = split.oot

        y_train = df_train["target"]
        y_test = df_test["target"]
        y_oot = df_oot["target"]

        scorecard = PDScorecard(
            pdo=cfg.scorecard.pdo,
            base_score=cfg.scorecard.base_score,
            base_odds=cfg.scorecard.base_odds,
        )
        scorecard.fit(df_train, y_train, df_test, y_test)

        pd_train = scorecard.predict_proba(df_train)
        pd_oot = scorecard.predict_proba(df_oot)

        disc = compute_discrimination(y_oot.values, pd_oot)
        cal = compute_calibration(y_oot.values, pd_oot)
        psi = compute_psi(pd_train, pd_oot)

        ginis.append(disc["gini"])
        aucs.append(disc["auc"])
        briers.append(cal["brier_score"])
        psis.append(psi)

    ginis_arr = np.array(ginis)
    aucs_arr = np.array(aucs)
    briers_arr = np.array(briers)
    psis_arr = np.array(psis)

    assert ginis_arr.std() < 0.10, f"Gini std too high: {ginis_arr.std():.4f}"
    assert aucs_arr.std() < 0.10, f"AUC std too high: {aucs_arr.std():.4f}"
    assert briers_arr.std() < 0.02, f"Brier std too high: {briers_arr.std():.4f}"
    assert psis_arr.std() < 0.10, f"PSI std too high: {psis_arr.std():.4f}"

