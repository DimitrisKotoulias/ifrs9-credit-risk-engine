"""Main data loading and preparation pipeline.

Dispatches to real Lending Club CSV loader or synthetic generator based on config.
Applies target definition, leakage filter, and OOT split.

Usage:
    python -m credit_risk.data.loader
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from credit_risk.data.leakage import filter_origination_features, log_leakage_policy
from credit_risk.data.split import DataSplit, time_split
from credit_risk.data.target import define_target
from credit_risk.utils.config import Config, load_config

logger = logging.getLogger(__name__)

# Columns to keep after leakage filter for PD model (informational; actual
# filter uses deny-list pattern, not this allow-list)
_DTYPE_MAP: dict[str, type] = {
    "loan_amnt": float,
    "funded_amnt": float,
    "int_rate": float,
    "installment": float,
    "annual_inc": float,
    "dti": float,
    "delinq_2yrs": float,
    "open_acc": float,
    "pub_rec": float,
    "revol_bal": float,
    "revol_util": float,
    "total_acc": float,
    "inq_last_6mths": float,
    "mths_since_last_delinq": float,
    "fico_range_low": float,
    "fico_range_high": float,
}


def _load_real(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real Lending Club CSVs with minimal dtype coercion."""
    raw_dir = Path(cfg.data.raw_dir)
    accepted_path = raw_dir / cfg.data.accepted_file
    rejected_path = raw_dir / cfg.data.rejected_file

    if not accepted_path.exists():
        raise FileNotFoundError(
            f"Accepted loans file not found: {accepted_path}\n"
            "Run: make data-download\n"
            "Or set data.source: synthetic in config/config.yaml to use generated data."
        )

    logger.info("Loading accepted loans from %s (chunked)...", accepted_path)
    # Lending Club CSV has a trailing 'Notes offered' text row — skip with low_memory=False
    accepted = pd.read_csv(
        accepted_path,
        low_memory=False,
        na_values=["n/a", "N/A", "NA", ""],
    )
    # Drop completely empty rows (last 2 rows of Lending Club export)
    accepted = accepted.dropna(how="all").reset_index(drop=True)

    logger.info("Loaded %d accepted loans.", len(accepted))

    rejected: pd.DataFrame | None = None
    if rejected_path.exists():
        logger.info("Loading rejected loans from %s...", rejected_path)
        rejected = pd.read_csv(rejected_path, low_memory=False, na_values=["n/a", "N/A", "NA", ""])
        rejected = rejected.dropna(how="all").reset_index(drop=True)
        logger.info("Loaded %d rejected loans.", len(rejected))
    else:
        logger.warning("Rejected loans file not found at %s; skipping.", rejected_path)
        rejected = pd.DataFrame()

    return accepted, rejected


def _load_synthetic(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic Lending Club data."""
    from credit_risk.data.synthetic import generate_accepted, generate_rejected  # noqa: PLC0415

    logger.info(
        "Generating synthetic data (n=%d, seed=%d)...",
        cfg.data.synthetic_n_loans, cfg.random_seed,
    )
    accepted = generate_accepted(n_loans=cfg.data.synthetic_n_loans, seed=cfg.random_seed)
    rejected = generate_rejected(n_loans=cfg.data.synthetic_n_loans // 3, seed=cfg.random_seed + 1)
    return accepted, rejected


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns, handle % signs in rate fields."""
    out = df.copy()

    # int_rate comes as '12.50%' in real data
    if "int_rate" in out.columns and out["int_rate"].dtype == object:
        out["int_rate"] = pd.to_numeric(
            out["int_rate"].astype(str).str.replace("%", "").str.strip(),
            errors="coerce",
        ) / 100.0

    # revol_util similarly
    if "revol_util" in out.columns and out["revol_util"].dtype == object:
        out["revol_util"] = pd.to_numeric(
            out["revol_util"].astype(str).str.replace("%", "").str.strip(),
            errors="coerce",
        )

    # term: ' 36 months' → 36
    if "term" in out.columns and out["term"].dtype == object:
        out["term"] = pd.to_numeric(
            out["term"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )

    return out

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced credit risk underwriting features for model enhancement."""
    import numpy as np
    out = df.copy()

    def get_series(col, default_val=np.nan):
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        else:
            return pd.Series(default_val, index=out.index)

    # 1. Continuous FICO low
    fico_low = get_series("fico_range_low", 690.0).fillna(690.0)

    # 2. Debt Burden Ratios
    annual_inc = get_series("annual_inc", 50000.0).fillna(50000.0).clip(lower=1.0)
    loan_amnt = get_series("loan_amnt", 0.0).fillna(0.0)
    revol_bal = get_series("revol_bal", 0.0).fillna(0.0)
    
    out["loan_to_income"] = loan_amnt / annual_inc
    out["revol_bal_to_income"] = revol_bal / annual_inc
    out["total_debt_to_income"] = (loan_amnt + revol_bal) / annual_inc

    # 3. Credit Utilization Interactions
    revol_util = get_series("revol_util", 50.0).fillna(50.0)
    open_acc = get_series("open_acc", 10.0).fillna(10.0).clip(lower=1.0)
    total_acc = get_series("total_acc", 20.0).fillna(20.0).clip(lower=1.0)
    
    out["revol_util_x_open_acc"] = (revol_util / 100.0) * open_acc
    out["revol_util_x_total_acc"] = (revol_util / 100.0) * total_acc
    out["util_per_account"] = (revol_util / 100.0) / open_acc

    # 4. Delinquency Severity Score
    delinq_2yrs = get_series("delinq_2yrs", 0.0).fillna(0.0)
    pub_rec = get_series("pub_rec", 0.0).fillna(0.0)
    pub_rec_bankruptcies = get_series("pub_rec_bankruptcies", 0.0).fillna(0.0)
    
    out["delinq_severity"] = delinq_2yrs * 1.0 + pub_rec * 2.0 + pub_rec_bankruptcies * 3.0

    # 5. Credit Age Features
    if "earliest_cr_line" in out.columns and "issue_d" in out.columns:
        issue_d_dt = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce").fillna(pd.Timestamp("2015-01-01"))
        earliest_cr_dt = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce").fillna(pd.Timestamp("2000-01-01"))
        out["credit_age_months"] = ((issue_d_dt - earliest_cr_dt).dt.days / 30.44).fillna(180.0).clip(lower=0.0)
        out["credit_age_years"] = out["credit_age_months"] / 12.0
        out["avg_account_age"] = out["credit_age_months"] / total_acc
    else:
        out["credit_age_months"] = 180.0
        out["credit_age_years"] = 15.0
        out["avg_account_age"] = 9.0

    # 6. Recent Credit Behavior
    inq_last_6mths = get_series("inq_last_6mths", 0.0).fillna(0.0)
    acc_open_past_24mths = get_series("acc_open_past_24mths", 2.0).fillna(2.0)
    
    out["inq_last_6mths_per_month"] = inq_last_6mths / 6.0
    out["new_accounts_ratio"] = acc_open_past_24mths / total_acc

    # 7. Income Stability Proxies
    emp_len_str = out["emp_length"] if "emp_length" in out.columns else pd.Series(["0"] * len(out), index=out.index)
    emp_len_str = emp_len_str.astype(str)
    emp_len_num = pd.to_numeric(emp_len_str.str.extract(r"(\d+)")[0], errors="coerce").fillna(0.0)
    out["income_per_emp_year"] = annual_inc / (emp_len_num + 1.0)
    out["log_annual_inc"] = np.log1p(annual_inc)

    return out


def load_and_prepare(cfg: Config | None = None) -> tuple[DataSplit, pd.DataFrame]:
    """Full data loading, cleaning, target definition, leakage filter, OOT split.

    Returns
    -------
    tuple[DataSplit, pd.DataFrame]
        (DataSplit with train/test/oot DataFrames, rejected loans DataFrame)
    """
    if cfg is None:
        cfg = load_config()

    log_leakage_policy(cfg.leakage)

    if cfg.data.source == "real":
        accepted, rejected = _load_real(cfg)
    else:
        accepted, rejected = _load_synthetic(cfg)

    accepted = _clean_numeric_columns(accepted)
    accepted = add_engineered_features(accepted)

    # Deterministic sorting to ensure row indexing remains identical across different OS
    if "id" in accepted.columns:
        accepted = accepted.sort_values(by="id").reset_index(drop=True)
    else:
        sort_cols = [col for col in ["issue_d", "funded_amnt", "loan_amnt"] if col in accepted.columns]
        if sort_cols:
            accepted = accepted.sort_values(by=sort_cols).reset_index(drop=True)

    if len(rejected) > 0:
        if "id" in rejected.columns:
            rejected = rejected.sort_values(by="id").reset_index(drop=True)
        else:
            sort_cols_rej = [col for col in ["application_date", "loan_amnt"] if col in rejected.columns]
            if sort_cols_rej:
                rejected = rejected.sort_values(by=sort_cols_rej).reset_index(drop=True)

    # Target definition
    accepted = define_target(accepted, cfg.target)

    # Preserve pre-leakage data for LGD/EAD model fitting (they need
    # post-origination columns like 'recoveries' and 'total_pymnt')
    accepted_pre_leakage = accepted.copy()

    # Leakage filter (applied AFTER target definition so loan_status is still present
    # for target coding, but removed from features)
    features_df = filter_origination_features(accepted, cfg.leakage)

    # OOT split
    split = time_split(features_df, cfg.split, seed=cfg.random_seed)
    split.full_accepted = accepted_pre_leakage
    logger.info("Data preparation complete. %s", split)

    return split, rejected


if __name__ == "__main__":
    import logging as _logging

    from credit_risk.utils.logging import setup_logging

    setup_logging(_logging.INFO)
    cfg = load_config()
    split, rejected = load_and_prepare(cfg)
    logger.info("Split summary: %s", split)
    logger.info("Rejected loans: %d rows", len(rejected))
