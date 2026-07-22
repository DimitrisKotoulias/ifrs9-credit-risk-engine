"""Tests for data engineering: target, leakage, split, synthetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.data.leakage import filter_origination_features
from credit_risk.data.split import time_split
from credit_risk.data.target import TARGET_COL, define_target
from credit_risk.utils.config import LeakageConfig, SplitConfig, TargetConfig


# ── Target definition ──────────────────────────────────────────────────────────

class TestDefineTarget:
    def _cfg(self) -> TargetConfig:
        return TargetConfig(
            bad_statuses=["Charged Off", "Default", "Late (31-120 days)"],
            good_statuses=["Fully Paid"],
        )

    def test_bad_mapped_to_1(self) -> None:
        df = pd.DataFrame({"loan_status": ["Charged Off", "Default", "Late (31-120 days)"]})
        out = define_target(df, self._cfg())
        assert (out[TARGET_COL] == 1).all()

    def test_good_mapped_to_0(self) -> None:
        df = pd.DataFrame({"loan_status": ["Fully Paid", "Fully Paid"]})
        out = define_target(df, self._cfg())
        assert (out[TARGET_COL] == 0).all()

    def test_excluded_rows_dropped(self) -> None:
        df = pd.DataFrame({
            "loan_status": ["Fully Paid", "Current", "In Grace Period", "Charged Off"]
        })
        out = define_target(df, self._cfg())
        assert len(out) == 2  # Current and In Grace Period excluded

    def test_all_spec_bad_statuses_covered(self) -> None:
        bad_statuses = [
            "Charged Off",
            "Default",
            "Does not meet the credit policy. Status:Charged Off",
            "Late (31-120 days)",
        ]
        cfg = TargetConfig(bad_statuses=bad_statuses, good_statuses=["Fully Paid"])
        df = pd.DataFrame({"loan_status": bad_statuses})
        out = define_target(df, cfg)
        assert (out[TARGET_COL] == 1).all()
        assert len(out) == len(bad_statuses)

    def test_all_spec_good_statuses_covered(self) -> None:
        good_statuses = [
            "Fully Paid",
            "Does not meet the credit policy. Status:Fully Paid",
        ]
        cfg = TargetConfig(bad_statuses=["Charged Off"], good_statuses=good_statuses)
        df = pd.DataFrame({"loan_status": good_statuses})
        out = define_target(df, cfg)
        assert (out[TARGET_COL] == 0).all()

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"status": ["Fully Paid"]})
        with pytest.raises(ValueError, match="loan_status"):
            define_target(df, self._cfg())

    def test_overlap_raises(self) -> None:
        cfg = TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Charged Off"])
        df = pd.DataFrame({"loan_status": ["Charged Off"]})
        with pytest.raises(ValueError, match="both bad and good"):
            define_target(df, cfg)


# ── Leakage filter ─────────────────────────────────────────────────────────────

class TestLeakageFilter:
    def _cfg(self) -> LeakageConfig:
        return LeakageConfig(
            deny_list=[
                "recoveries", "collection_recovery_fee", "total_pymnt",
                "total_rec_prncp", "last_pymnt_d", "out_prncp",
                "funded_amnt_inv", "debt_settlement_flag", "loan_status",
            ],
            allow_overrides=[],
        )

    def test_explicit_deny_columns_removed(self) -> None:
        df = pd.DataFrame({
            "loan_amnt": [1000.0],
            "recoveries": [50.0],
            "collection_recovery_fee": [5.0],
            "total_pymnt": [1100.0],
            "funded_amnt_inv": [990.0],
        })
        out = filter_origination_features(df, self._cfg())
        for col in ["recoveries", "collection_recovery_fee", "total_pymnt", "funded_amnt_inv"]:
            assert col not in out.columns, f"Column {col!r} should have been removed"

    def test_prefix_variants_removed(self) -> None:
        """total_pymnt deny-list should also match total_pymnt_inv, total_pymnt_amnt."""
        df = pd.DataFrame({
            "loan_amnt": [1000.0],
            "total_pymnt_inv": [900.0],
            "total_rec_prncp": [800.0],
            "out_prncp_inv": [200.0],
        })
        out = filter_origination_features(df, self._cfg())
        for col in ["total_pymnt_inv", "total_rec_prncp", "out_prncp_inv"]:
            assert col not in out.columns

    def test_origination_features_preserved(self) -> None:
        safe_cols = ["loan_amnt", "int_rate", "dti", "grade", "annual_inc"]
        df = pd.DataFrame({col: [1.0] for col in safe_cols})
        out = filter_origination_features(df, self._cfg())
        for col in safe_cols:
            assert col in out.columns

    def test_allow_override_preserved(self) -> None:
        cfg = LeakageConfig(
            deny_list=["loan_status"],
            allow_overrides=["loan_status"],
        )
        df = pd.DataFrame({"loan_status": ["Fully Paid"], "grade": ["A"]})
        out = filter_origination_features(df, cfg)
        assert "loan_status" in out.columns

    def test_all_spec_deny_columns_removed(self, small_accepted: pd.DataFrame) -> None:
        """All columns from spec §4.2 should be filtered out."""
        cfg = self._cfg()
        out = filter_origination_features(small_accepted, cfg)
        for col in cfg.deny_list:
            assert col not in out.columns, f"Spec deny-list column {col!r} still present"


# ── Time split ─────────────────────────────────────────────────────────────────

class TestTimeSplit:
    def _cfg(self) -> SplitConfig:
        return SplitConfig(
            train_cutoff="2015-01-01",
            oot_cutoff="2016-01-01",
            holdout_frac=0.20,
        )

    def test_no_date_overlap_train_oot(self, small_accepted: pd.DataFrame) -> None:
        """No loan origination date should appear in both train/test and OOT sets."""
        from credit_risk.data.split import parse_issue_date

        split = time_split(small_accepted, self._cfg(), seed=42)
        if len(split.oot) == 0:
            return  # nothing to check if OOT is empty (small fixture may not span dates)
        train_test_dates = set(parse_issue_date(split.train).astype(str)) | set(
            parse_issue_date(split.test).astype(str)
        )
        oot_dates = set(parse_issue_date(split.oot).astype(str))
        # OOT dates must be strictly after oot_cutoff; train/test before train_cutoff
        # so they cannot overlap
        oot_cutoff = pd.Timestamp("2016-01-01")
        train_cutoff = pd.Timestamp("2015-01-01")
        assert all(
            pd.Timestamp(d) >= oot_cutoff for d in oot_dates if pd.notna(d)
        ), "OOT contains dates before oot_cutoff"
        assert all(
            pd.Timestamp(d) < train_cutoff for d in train_test_dates if pd.notna(d)
        ), "Train/test contains dates at or after train_cutoff"

    def test_train_before_cutoff(self, small_accepted: pd.DataFrame) -> None:
        from credit_risk.data.split import parse_issue_date

        split = time_split(small_accepted, self._cfg(), seed=42)
        cutoff = pd.Timestamp("2015-01-01")
        train_dates = parse_issue_date(split.train)
        assert (train_dates < cutoff).all(), "Train set contains rows after train_cutoff"

    def test_oot_after_cutoff(self, small_accepted: pd.DataFrame) -> None:
        from credit_risk.data.split import parse_issue_date

        split = time_split(small_accepted, self._cfg(), seed=42)
        oot_cutoff = pd.Timestamp("2016-01-01")
        oot_dates = parse_issue_date(split.oot)
        if len(oot_dates) > 0:
            assert (oot_dates >= oot_cutoff).all(), "OOT set has rows before oot_cutoff"

    def test_holdout_fraction(self, small_accepted: pd.DataFrame) -> None:
        split = time_split(small_accepted, self._cfg(), seed=42)
        total_train_period = len(split.train) + len(split.test)
        if total_train_period > 0:
            frac = len(split.test) / total_train_period
            assert abs(frac - 0.20) < 0.05, f"Holdout fraction {frac:.2f} far from 0.20"

    def test_invalid_cutoffs_raise(self, small_accepted: pd.DataFrame) -> None:
        bad_cfg = SplitConfig(
            train_cutoff="2016-01-01",
            oot_cutoff="2015-01-01",
            holdout_frac=0.20,
        )
        with pytest.raises(ValueError, match="train_cutoff"):
            time_split(small_accepted, bad_cfg)


# ── Synthetic data ─────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_accepted_default_rate_nonzero(self, small_accepted: pd.DataFrame) -> None:
        """Synthetic data must have a realistic default rate."""
        from credit_risk.data.target import define_target

        cfg = TargetConfig(
            bad_statuses=["Charged Off"],
            good_statuses=["Fully Paid"],
        )
        df = define_target(small_accepted, cfg)
        dr = df[TARGET_COL].mean()
        assert 0.05 < dr < 0.50, f"Default rate {dr:.2%} is outside realistic range"

    def test_embedded_pd_signal_auc(self, small_accepted: pd.DataFrame) -> None:
        """Grade should predict default with AUC > 0.60 on synthetic data."""
        from credit_risk.data.target import define_target

        cfg = TargetConfig(bad_statuses=["Charged Off"], good_statuses=["Fully Paid"])
        df = define_target(small_accepted, cfg)
        grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        score = df["grade"].map(grade_map).fillna(4).values
        auc = roc_auc_score(df[TARGET_COL], score)
        assert auc > 0.60, f"Grade AUC {auc:.3f} — synthetic PD signal too weak"

    def test_rejected_has_no_loan_status(self, small_rejected: pd.DataFrame) -> None:
        assert "loan_status" not in small_rejected.columns

    def test_schema_consistency(self, small_accepted: pd.DataFrame) -> None:
        expected_cols = [
            "loan_amnt", "funded_amnt", "int_rate", "grade", "dti",
            "annual_inc", "term", "issue_d", "loan_status", "recoveries",
        ]
        for col in expected_cols:
            assert col in small_accepted.columns, f"Expected column {col!r} missing"
