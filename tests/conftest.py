"""Shared pytest fixtures — all use synthetic data so no CSVs required."""

from __future__ import annotations

# Importing the package installs the optbinning/sklearn compatibility shim. The tests
# deliberately do NOT patch anything themselves: a test-only patch would let the suite
# pass against a production path that is broken (FLAWS-N22).
import credit_risk  # noqa: F401
import numpy as np
import pandas as pd
import pytest

RNG_SEED = 42


@pytest.fixture(scope="function")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="function")
def small_accepted(rng: np.random.Generator) -> pd.DataFrame:
    """Mini accepted-loans dataframe (500 rows) with realistic schema."""
    n = 500
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    grade_arr = rng.choice(grades, size=n, p=[0.15, 0.20, 0.22, 0.18, 0.12, 0.08, 0.05])
    int_rate = np.clip(
        rng.normal(0.13, 0.06, n) + np.array([grades.index(g) * 0.02 for g in grade_arr]),
        0.05,
        0.35,
    )
    funded_amnt = rng.integers(1000, 35001, n).astype(float)
    dti = np.clip(rng.normal(18, 8, n), 0, 60)
    annual_inc = np.clip(rng.lognormal(10.8, 0.6, n), 15000, 500000)
    term = rng.choice([36, 60], size=n, p=[0.6, 0.4])

    # Embedded default signal
    log_odds = (
        -3.5
        + 0.5 * np.array([grades.index(g) for g in grade_arr]) / 3
        + 0.04 * int_rate * 100
        + 0.015 * dti
        - 0.0001 * annual_inc / 1000
    )
    pd_true = 1 / (1 + np.exp(-log_odds))
    is_bad = rng.random(n) < pd_true

    # Recoveries (post-origination, for LGD model)
    recoveries = np.where(is_bad, funded_amnt * np.clip(rng.beta(0.5, 2, n), 0, 1), 0.0)

    # Loan status
    status = np.where(is_bad, "Charged Off", "Fully Paid")

    # Issue date — spread 2010–2017
    days = rng.integers(0, 365 * 7, n)
    issue_d = pd.to_datetime("2010-01-01") + pd.to_timedelta(days, unit="D")
    issue_d_str = issue_d.strftime("%b-%Y")

    # Payment history. Without these columns `compute_realised_lgd` runs with
    # total_rec_prncp = 0 — so the principal-basis denominator that is the whole point of
    # that module is never exercised — and every months-on-book estimate falls back to
    # `term * 0.4`. Since the hazard model's person-period panel is now built from exactly
    # this duration proxy, a fixture without them silently tests the uncensored fallback
    # instead of the production path. Defaulters stop paying mid-term; survivors run on.
    installment = funded_amnt * int_rate / 12 / (1 - (1 + int_rate / 12) ** (-term))
    mob_obs = np.where(is_bad, rng.integers(6, 25, n).astype(float), term.astype(float))
    mob_obs = np.minimum(mob_obs, term)
    total_pymnt = installment * mob_obs
    # Charged-off loans repay only part of the principal before defaulting.
    principal_share = np.where(is_bad, rng.uniform(0.15, 0.75, n), 1.0)
    total_rec_prncp = funded_amnt * principal_share
    last_pymnt_d = (issue_d + pd.to_timedelta((mob_obs * 30.44).astype(int), unit="D")
                    ).strftime("%b-%Y")

    return pd.DataFrame(
        {
            "loan_amnt": funded_amnt,
            "funded_amnt": funded_amnt,
            "int_rate": int_rate,
            "grade": grade_arr,
            "sub_grade": [g + str(rng.integers(1, 6)) for g in grade_arr],
            "term": [f" {t} months" for t in term],
            "emp_length": rng.choice(
                ["< 1 year", "1 year", "2 years", "5 years", "10+ years"], size=n
            ),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], size=n),
            "annual_inc": annual_inc,
            "verification_status": rng.choice(
                ["Verified", "Source Verified", "Not Verified"], size=n
            ),
            "purpose": rng.choice(
                ["debt_consolidation", "credit_card", "home_improvement", "other"], size=n
            ),
            "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], size=n),
            "dti": dti,
            "delinq_2yrs": rng.integers(0, 4, n).astype(float),
            "open_acc": rng.integers(3, 25, n).astype(float),
            # Source of the `revol_util_x_new_acc` interaction. Without it the scorecard's
            # interaction step is a no-op in tests, so the train/serve skew that silently
            # neutralised that feature in production could not be reproduced.
            "acc_open_past_24mths": rng.integers(0, 12, n).astype(float),
            "pub_rec": rng.integers(0, 3, n).astype(float),
            "revol_util": np.clip(rng.normal(50, 25, n), 0, 100),
            "total_acc": rng.integers(5, 50, n).astype(float),
            "recoveries": recoveries,
            "issue_d": issue_d_str,
            "loan_status": status,
            "installment": installment,
            "total_pymnt": total_pymnt,
            "total_rec_prncp": total_rec_prncp,
            "last_pymnt_d": last_pymnt_d,
            "fico_range_low": np.clip(rng.normal(690, 50, n), 580, 850).astype(int),
            "fico_range_high": np.clip(rng.normal(694, 50, n), 584, 854).astype(int),
            "earliest_cr_line": rng.choice(
                ["Jan-2000", "Mar-2005", "Jun-1998", "Nov-2010"], size=n
            ),
            "inq_last_6mths": rng.integers(0, 6, n).astype(float),
            "mths_since_last_delinq": rng.choice(
                [np.nan, 12.0, 24.0, 36.0, 48.0], size=n
            ),
            "revol_bal": np.clip(rng.lognormal(8, 1.5, n), 0, 100000),
        }
    )


@pytest.fixture(scope="function")
def small_rejected(rng: np.random.Generator) -> pd.DataFrame:
    """Mini rejected-loans dataframe (200 rows)."""
    n = 200
    return pd.DataFrame(
        {
            "loan_amnt": rng.integers(1000, 35001, n).astype(float),
            "application_date": pd.to_datetime("2010-01-01")
            + pd.to_timedelta(rng.integers(0, 365 * 7, n), unit="D"),
            "title": rng.choice(["Debt consolidation", "Credit card", "Other"], size=n),
            "dti": np.clip(rng.normal(22, 10, n), 0, 70),
            "zip_code": rng.choice(["945xx", "100xx", "770xx"], size=n),
            "addr_state": rng.choice(["CA", "NY", "TX"], size=n),
            "emp_length": rng.choice(["< 1 year", "1 year", "2 years", "5 years"], size=n),
            "policy_code": "0",
            "annual_inc": np.clip(rng.lognormal(10.5, 0.7, n), 10000, 300000),
            "risk_score": np.clip(rng.normal(620, 60, n), 500, 800),
        }
    )
