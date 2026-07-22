"""Synthetic Lending Club data generator.

Produces dataframes with the same schema as the real Lending Club accepted/rejected
CSVs, with an embedded logistic PD signal so all downstream models are non-trivial.
Used for pytest fixtures and CI only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_GRADES = ["A", "B", "C", "D", "E", "F", "G"]
_GRADE_PROBS = [0.15, 0.20, 0.22, 0.18, 0.12, 0.08, 0.05]
_PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "medical", "other"]
_STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
_HOME = ["RENT", "MORTGAGE", "OWN"]
_EMP_LENGTHS = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "10+ years"]
_VERIF = ["Verified", "Source Verified", "Not Verified"]


def generate_accepted(n_loans: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic accepted loans with embedded default signal.

    Parameters
    ----------
    n_loans:
        Number of rows to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the key Lending Club accepted-loans schema.
    """
    rng = np.random.default_rng(seed)

    grade_idx = rng.choice(len(_GRADES), size=n_loans, p=_GRADE_PROBS)
    grades = np.array(_GRADES)[grade_idx]
    sub_grade_num = rng.integers(1, 6, size=n_loans)
    sub_grades = np.array([g + str(n) for g, n in zip(grades, sub_grade_num)])

    term = rng.choice([36, 60], size=n_loans, p=[0.60, 0.40])
    int_rate = np.clip(
        0.06 + 0.018 * grade_idx + rng.normal(0, 0.015, n_loans),
        0.05, 0.35,
    )
    funded_amnt = rng.choice(
        [1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 25000, 35000],
        size=n_loans,
    ).astype(float)
    annual_inc = np.exp(rng.normal(10.8, 0.6, n_loans)).clip(15_000, 500_000)
    dti = np.clip(rng.normal(17.5, 8.0, n_loans), 0, 60)
    fico_low = np.clip(rng.normal(700 - grade_idx * 12, 40, n_loans), 580, 850).astype(int)
    fico_high = (fico_low + rng.integers(2, 6, n_loans)).astype(int)
    delinq = rng.choice([0, 0, 0, 1, 1, 2, 3], size=n_loans).astype(float)
    open_acc = rng.integers(3, 30, n_loans).astype(float)
    pub_rec = rng.choice([0, 0, 0, 1, 2], size=n_loans, p=[0.7, 0.15, 0.1, 0.03, 0.02]).astype(float)
    revol_util = np.clip(rng.normal(50, 25, n_loans), 0, 100)
    revol_bal = np.exp(rng.normal(8, 1.5, n_loans)).clip(0, 100_000)
    total_acc = rng.integers(5, 55, n_loans).astype(float)
    inq_6m = rng.choice([0, 0, 1, 1, 2, 3, 4], size=n_loans).astype(float)
    mths_since_delinq: Any = np.where(
        delinq > 0,
        rng.choice([12.0, 24.0, 36.0, 48.0, 60.0], size=n_loans),
        np.nan,
    )

    # Issue date: 2010-01 to 2017-12
    days_offset = rng.integers(0, 365 * 8, n_loans)
    issue_dt = pd.to_datetime("2010-01-01") + pd.to_timedelta(days_offset, unit="D")
    issue_d_str = issue_dt.strftime("%b-%Y")

    earliest_cr = (issue_dt - pd.to_timedelta(rng.integers(365, 365 * 20, n_loans), unit="D")).strftime("%b-%Y")

    installment = (
        funded_amnt * (int_rate / 12) / (1 - (1 + int_rate / 12) ** (-term))
    )

    # Logistic default signal (using only origination features)
    log_odds = (
        -4.0
        + 0.55 * grade_idx / 3.0
        + 0.04 * (int_rate * 100)
        + 0.015 * dti
        - 0.0001 * annual_inc / 1000
        + 0.08 * delinq
        - 0.002 * (fico_low - 650)
        + 0.03 * inq_6m
    )
    pd_true = 1.0 / (1.0 + np.exp(-log_odds))
    is_bad = rng.random(n_loans) < pd_true

    # Recoveries: post-origination (used only for LGD model — not for PD features)
    cure = rng.random(n_loans) < 0.40
    severity = np.where(cure, 0.0, np.clip(rng.beta(0.5, 2, n_loans), 0, 1))
    recoveries = np.where(is_bad, funded_amnt * (1 - severity), 0.0)

    loan_status = np.where(is_bad, "Charged Off", "Fully Paid")

    logger.info(
        "Synthetic accepted: n=%d, default_rate=%.2f%%",
        n_loans,
        is_bad.mean() * 100,
    )

    return pd.DataFrame({
        "loan_amnt": funded_amnt,
        "funded_amnt": funded_amnt,
        "funded_amnt_inv": funded_amnt * rng.uniform(0.95, 1.0, n_loans),  # leakage test
        "term": [f" {t} months" for t in term],
        "int_rate": int_rate,
        "installment": installment,
        "grade": grades,
        "sub_grade": sub_grades,
        "emp_title": "Employee",
        "emp_length": rng.choice(_EMP_LENGTHS, size=n_loans),
        "home_ownership": rng.choice(_HOME, size=n_loans),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(_VERIF, size=n_loans),
        "issue_d": issue_d_str,
        "loan_status": loan_status,
        "purpose": rng.choice(_PURPOSES, size=n_loans),
        "title": "Loan",
        "addr_state": rng.choice(_STATES, size=n_loans),
        "dti": dti,
        "delinq_2yrs": delinq,
        "earliest_cr_line": earliest_cr,
        "fico_range_low": fico_low,
        "fico_range_high": fico_high,
        "inq_last_6mths": inq_6m,
        "mths_since_last_delinq": mths_since_delinq,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        # Post-origination features (must be in leakage deny-list)
        "recoveries": recoveries,
        "collection_recovery_fee": recoveries * 0.1,
        "total_pymnt": np.where(is_bad, funded_amnt * 0.3, funded_amnt * 1.1),
        "total_rec_prncp": np.where(is_bad, funded_amnt * 0.2, funded_amnt),
        "total_rec_int": funded_amnt * int_rate * term / 12 * rng.uniform(0.5, 1.0, n_loans),
        "last_pymnt_amnt": rng.uniform(0, 500, n_loans),
        "out_prncp": np.where(is_bad, funded_amnt * 0.5, 0.0),
        "debt_settlement_flag": np.where(is_bad & (rng.random(n_loans) < 0.1), "Y", "N"),
        "hardship_flag": "N",
    })


def generate_rejected(n_loans: int = 20_000, seed: int = 43) -> pd.DataFrame:
    """Generate synthetic rejected loans (no outcome labels).

    Parameters
    ----------
    n_loans:
        Number of rows.
    seed:
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame matching the Lending Club rejected-loans schema.
    """
    rng = np.random.default_rng(seed)

    days_offset = rng.integers(0, 365 * 8, n_loans)
    app_date = pd.to_datetime("2010-01-01") + pd.to_timedelta(days_offset, unit="D")

    # Rejected applicants tend to have worse risk profiles
    fico = np.clip(rng.normal(615, 60, n_loans), 500, 760).astype(int)
    dti = np.clip(rng.normal(28, 12, n_loans), 0, 70)
    annual_inc = np.exp(rng.normal(10.4, 0.7, n_loans)).clip(10_000, 300_000)

    logger.info("Synthetic rejected: n=%d", n_loans)

    return pd.DataFrame({
        "loan_amnt": rng.integers(500, 40_001, n_loans).astype(float),
        "application_date": app_date,
        "title": rng.choice(["Debt consolidation", "Credit card", "Home improvement", "Other"], size=n_loans),
        "risk_score": fico,
        "debt_to_income_ratio": dti,
        "zip_code": rng.choice(["945xx", "100xx", "770xx", "606xx", "900xx"], size=n_loans),
        "addr_state": rng.choice(_STATES, size=n_loans),
        "employment_length": rng.choice(_EMP_LENGTHS, size=n_loans),
        "policy_code": "0",
        "annual_inc": annual_inc,
    })
