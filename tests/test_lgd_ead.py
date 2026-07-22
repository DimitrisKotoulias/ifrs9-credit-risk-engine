"""Tests for LGD and EAD models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.models.ead import EADModel, amortisation_factor
from credit_risk.models.lgd import LGDModel, compute_realised_lgd


class TestRealisedLGD:
    def test_lgd_in_0_1(self) -> None:
        df = pd.DataFrame({
            "recoveries": [0.0, 500.0, 1000.0, 1100.0],
            "funded_amnt": [1000.0, 1000.0, 1000.0, 1000.0],
        })
        lgd = compute_realised_lgd(df)
        assert (lgd >= 0.0).all() and (lgd <= 1.0).all()

    def test_lgd_zero_recovery_is_one(self) -> None:
        df = pd.DataFrame({"recoveries": [0.0], "funded_amnt": [5000.0]})
        lgd = compute_realised_lgd(df)
        assert float(lgd.iloc[0]) == pytest.approx(1.0)

    def test_lgd_full_recovery_is_zero(self) -> None:
        df = pd.DataFrame({"recoveries": [5000.0], "funded_amnt": [5000.0]})
        lgd = compute_realised_lgd(df)
        assert float(lgd.iloc[0]) == pytest.approx(0.0)

    def test_lgd_clipped_at_0_and_1(self) -> None:
        df = pd.DataFrame({
            "recoveries": [-500.0, 9999.0],
            "funded_amnt": [1000.0, 1000.0],
        })
        lgd = compute_realised_lgd(df)
        assert float(lgd.iloc[0]) == pytest.approx(1.0)  # negative recovery → clipped to LGD=1
        assert float(lgd.iloc[1]) == pytest.approx(0.0)  # over-recovery → clipped to LGD=0

    def test_lgd_with_total_rec_prncp(self) -> None:
        # Loan with 1000 funded, 200 principal already repaid before default, 400 recovered after default.
        # EAD at default = 1000 - 200 = 800.
        # Realised LGD = 1 - (400 / 800) = 0.50.
        df = pd.DataFrame({
            "recoveries": [400.0],
            "funded_amnt": [1000.0],
            "total_rec_prncp": [200.0],
        })
        lgd = compute_realised_lgd(df)
        assert float(lgd.iloc[0]) == pytest.approx(0.50)

    def test_compute_realised_lgd_net_recovery(self):
        import pandas as pd
        from credit_risk.models.lgd import compute_realised_lgd
        df = pd.DataFrame({
            "funded_amnt": [10_000.0], "total_rec_prncp": [3_000.0],
            "recoveries": [500.0], "collection_recovery_fee": [50.0],
        })
        result = compute_realised_lgd(df)
        assert abs(result.iloc[0] - (1 - 450/7000)) < 1e-6

    def test_compute_realised_lgd_over_recovery(self):
        import pandas as pd
        from credit_risk.models.lgd import compute_realised_lgd
        df = pd.DataFrame({
            "funded_amnt": [5_000.0], "total_rec_prncp": [4_500.0],
            "recoveries": [600.0], "collection_recovery_fee": [0.0],
        })
        result = compute_realised_lgd(df)
        assert result.iloc[0] == 0.0

    def test_compute_realised_lgd_ignores_interest_in_total_pymnt(self):
        # funded=10000, principal recovered=3000 -> EAD=7000.
        # total_pymnt=5000 includes 2000 of interest/fees on top of the 3000
        # principal - that interest must NOT reduce LGD, since LGD is a
        # principal-basis measure. recoveries=500, fee=50 -> net_recoveries=450.
        # Expected LGD = 1 - 450/7000, same as the no-total_pymnt case.
        df = pd.DataFrame({
            "funded_amnt": [10_000.0],
            "total_rec_prncp": [3_000.0],
            "total_pymnt": [5_000.0],
            "recoveries": [500.0],
            "collection_recovery_fee": [50.0],
        })
        result = compute_realised_lgd(df)
        assert abs(result.iloc[0] - (1 - 450 / 7000)) < 1e-6


class TestAmortisationFactor:
    def test_factor_at_t0_is_one(self) -> None:
        f = amortisation_factor(0, 36, 0.12)
        assert float(f) == pytest.approx(1.0, abs=1e-6)

    def test_factor_monotone_decreasing(self) -> None:
        t_vals = np.arange(0, 37)
        factors = amortisation_factor(t_vals, 36, 0.12)
        assert np.all(np.diff(factors) <= 0), "Amortisation factor must decrease with time"

    def test_factor_near_zero_at_maturity(self) -> None:
        f = amortisation_factor(36, 36, 0.12)
        assert float(f) == pytest.approx(0.0, abs=1e-6)

    def test_zero_rate_linear(self) -> None:
        f = amortisation_factor(18, 36, 0.0)
        assert float(f) == pytest.approx(0.5, abs=1e-6)

    def test_factor_in_0_1(self) -> None:
        t = np.random.default_rng(42).integers(0, 61, 100)
        f = amortisation_factor(t, 60, 0.15)
        assert (f >= 0.0).all() and (f <= 1.0).all()

    def test_term_60_higher_outstanding_than_36_at_same_mob(self) -> None:
        """60-month loan has more outstanding principal than 36-month at same MOB."""
        f36 = amortisation_factor(12, 36, 0.10)
        f60 = amortisation_factor(12, 60, 0.10)
        assert f60 > f36, "60-month loan should have more remaining principal after 12 months"


class TestEADModel:
    def test_ead_nonnegative(self, small_accepted: pd.DataFrame) -> None:
        model = EADModel()
        model.fit(small_accepted)
        ead = model.predict(small_accepted)
        assert (ead >= 0.0).all()

    def test_ead_not_exceeds_funded_amnt(self, small_accepted: pd.DataFrame) -> None:
        model = EADModel()
        model.fit(small_accepted)
        ead = model.predict(small_accepted)
        funded = pd.to_numeric(small_accepted["funded_amnt"], errors="coerce")
        assert (ead <= funded * 1.01).all(), "EAD should not exceed funded amount"

    def test_ead_length_matches_input(self, small_accepted: pd.DataFrame) -> None:
        model = EADModel()
        model.fit(small_accepted)
        ead = model.predict(small_accepted)
        assert len(ead) == len(small_accepted)


def test_ead_zero_rate_no_division_error():
    import pandas as pd
    from credit_risk.models.ead import EADModel
    df = pd.DataFrame({"funded_amnt": [10_000.0], "int_rate": [0.0], "term": [" 36 months"], "mob": [12]})
    model = EADModel()
    model.fit(df)
    ead = model.predict(df)
    assert 0 < ead.iloc[0] <= 10_000.0


class TestLGDModel:
    def test_lgd_predictions_in_0_1(self, small_accepted: pd.DataFrame) -> None:
        from credit_risk.data.target import TARGET_COL, define_target
        from credit_risk.utils.config import TargetConfig

        cfg = TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"])
        df = define_target(small_accepted, cfg)
        defaults = df[df[TARGET_COL] == 1].copy()
        if len(defaults) < 20:
            pytest.skip("Not enough defaults in small fixture.")

        model = LGDModel(downturn_percentile=90.0)
        model.fit(defaults)
        preds = model.predict(defaults)
        assert (preds >= 0.0).all() and (preds <= 1.0).all()

    def test_downturn_lgd_geq_mean_lgd(self, small_accepted: pd.DataFrame) -> None:
        from credit_risk.data.target import TARGET_COL, define_target
        from credit_risk.utils.config import TargetConfig

        cfg = TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"])
        df = define_target(small_accepted, cfg)
        defaults = df[df[TARGET_COL] == 1].copy()
        if len(defaults) < 20:
            pytest.skip("Not enough defaults in small fixture.")

        model = LGDModel(downturn_percentile=90.0)
        model.fit(defaults)
        assert model.downturn_lgd >= model.mean_lgd, (
            f"Downturn LGD ({model.downturn_lgd:.4f}) must be ≥ mean LGD ({model.mean_lgd:.4f})"
        )

    def test_two_stage_prediction_decomposes(self, small_accepted: pd.DataFrame) -> None:
        """Predicted LGD = P(loss) × E[severity|loss] — verify decomposition is bounded."""
        from credit_risk.data.target import TARGET_COL, define_target
        from credit_risk.utils.config import TargetConfig

        cfg = TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"])
        df = define_target(small_accepted, cfg)
        defaults = df[df[TARGET_COL] == 1].copy()
        if len(defaults) < 20:
            pytest.skip("Not enough defaults in small fixture.")

        model = LGDModel()
        model.fit(defaults)
        preds = model.predict(defaults)
        # Each output is P(loss) × severity, both ∈ [0, 1], so product ∈ [0, 1]
        assert (preds >= 0.0).all()
        assert (preds <= 1.0).all()
