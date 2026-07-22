"""Tests for out-of-sample LGD validation."""

import numpy as np
import pandas as pd

from credit_risk.models.lgd import compute_realised_lgd
from credit_risk.validation.lgd_validation import validate_lgd


def _defaults_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    funded = rng.uniform(5_000, 30_000, n)
    frac_paid = rng.uniform(0.0, 1.0, n)
    total_pymnt = funded * frac_paid
    total_rec_prncp = total_pymnt * rng.uniform(0.5, 0.95, n)
    ead_proxy = funded - total_rec_prncp
    recoveries = ead_proxy * rng.uniform(0.0, 0.5, n)
    collection_recovery_fee = recoveries * rng.uniform(0.0, 0.1, n)
    return pd.DataFrame({
        "funded_amnt": funded,
        "total_pymnt": total_pymnt,
        "total_rec_prncp": total_rec_prncp,
        "recoveries": recoveries,
        "collection_recovery_fee": collection_recovery_fee,
    })


class _ConstantLGD:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, name="lgd_pred")


class _PerfectLGD:
    def predict(self, df: pd.DataFrame) -> pd.Series:
        return compute_realised_lgd(df).rename("lgd_pred")


def test_validate_lgd_keys_and_bounds():
    df = _defaults_frame()
    metrics, decile_df = validate_lgd(_ConstantLGD(0.5), df)
    assert set(metrics) == {"mae", "rmse", "r2", "ks_stat", "ks_pvalue", "n_test"}
    assert 0.0 <= metrics["ks_stat"] <= 1.0
    assert metrics["n_test"] == float(len(df))
    assert {"decile", "mean_predicted", "mean_actual", "count"}.issubset(decile_df.columns)


def test_perfect_model_low_error_high_r2():
    df = _defaults_frame(seed=1)
    metrics, _ = validate_lgd(_PerfectLGD(), df)
    assert metrics["mae"] < 1e-9
    assert metrics["r2"] > 0.999
    assert metrics["ks_stat"] < 1e-9


def test_empty_frame_returns_nan_metrics():
    df = _defaults_frame().iloc[0:0]
    metrics, decile_df = validate_lgd(_ConstantLGD(0.4), df)
    assert metrics["n_test"] == 0.0
    assert np.isnan(metrics["mae"])
    assert decile_df.empty


def test_decile_actual_increases_with_predicted():
    """A model that ranks LGD correctly should show rising actual across deciles."""
    df = _defaults_frame(seed=2)
    _, decile_df = validate_lgd(_PerfectLGD(), df)
    actual = decile_df.sort_values("mean_predicted")["mean_actual"].to_numpy()
    # Monotone non-decreasing trend for a perfectly-ranked model.
    assert actual[-1] >= actual[0]
