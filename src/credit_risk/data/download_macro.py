"""Macroeconomic data downloader and generator.

Fetches historical US macroeconomic quarterly data from the official FRED
(St. Louis Fed) REST API and falls back to a high-fidelity real-to-history
offline dataset if the live pull fails (e.g. no FRED_API_KEY, no network).
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"

# FRED series IDs -> destination column name in macro_quarterly.csv.
# CSUSHPISA (Case-Shiller US National Home Price Index, seasonally adjusted)
# is used rather than the NSA variant so its QoQ growth rate is directly
# comparable to the already-seasonally-adjusted UNRATE/CPIAUCSL series.
_FRED_SERIES = {
    "UNRATE": "UNRATE",
    "GDP": "GDP",
    "CPIAUCSL": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "CSUSHPISA": "HPI",
}

# High-fidelity real historical quarterly US macro data (2007Q1 - 2018Q4)
# Source: St. Louis Fed (FRED)
_REAL_HISTORY = {
    "quarter": [
        "2007Q1", "2007Q2", "2007Q3", "2007Q4",
        "2008Q1", "2008Q2", "2008Q3", "2008Q4",
        "2009Q1", "2009Q2", "2009Q3", "2009Q4",
        "2010Q1", "2010Q2", "2010Q3", "2010Q4",
        "2011Q1", "2011Q2", "2011Q3", "2011Q4",
        "2012Q1", "2012Q2", "2012Q3", "2012Q4",
        "2013Q1", "2013Q2", "2013Q3", "2013Q4",
        "2014Q1", "2014Q2", "2014Q3", "2014Q4",
        "2015Q1", "2015Q2", "2015Q3", "2015Q4",
        "2016Q1", "2016Q2", "2016Q3", "2016Q4",
        "2017Q1", "2017Q2", "2017Q3", "2017Q4",
        "2018Q1", "2018Q2", "2018Q3", "2018Q4"
    ],
    "UNRATE": [
        4.5, 4.5, 4.7, 4.8,
        5.0, 5.4, 6.0, 6.9,
        8.3, 9.3, 9.6, 9.9,
        9.8, 9.6, 9.5, 9.5,
        9.0, 9.1, 9.0, 8.6,
        8.3, 8.2, 8.0, 7.8,
        7.7, 7.5, 7.2, 6.9,
        6.7, 6.2, 6.1, 5.7,
        5.5, 5.4, 5.1, 5.0,
        4.9, 4.9, 4.9, 4.7,
        4.6, 4.4, 4.3, 4.1,
        4.0, 3.9, 3.8, 3.8
    ],
    "GDP_growth": [
        0.3, 0.8, 0.6, 0.5,
        -0.4, 0.5, -0.5, -2.1,
        -1.1, -0.2, 0.3, 1.0,
        0.5, 0.9, 0.7, 0.6,
        -0.3, 0.7, 0.3, 1.1,
        0.8, 0.5, 0.2, 0.1,
        0.9, 0.2, 0.8, 0.8,
        -0.3, 1.3, 1.2, 0.6,
        0.8, 0.7, 0.4, 0.2,
        0.6, 0.4, 0.7, 0.5,
        0.5, 0.6, 0.7, 0.9,
        0.9, 0.7, 0.8, 0.3
    ],
    "CPI_inflation": [
        0.6, 0.7, 0.4, 0.9,
        0.8, 1.0, -0.2, -2.1,
        0.3, 0.3, 0.4, 0.3,
        0.1, -0.1, 0.2, 0.4,
        0.6, 0.7, 0.4, -0.1,
        0.5, 0.2, 0.4, 0.3,
        0.3, 0.1, 0.3, 0.2,
        0.4, 0.5, 0.1, -0.2,
        -0.4, 0.5, 0.1, -0.1,
        -0.1, 0.6, 0.3, 0.4,
        0.6, 0.2, 0.5, 0.5,
        0.6, 0.4, 0.2, 0.0
    ],
    "FEDFUNDS": [
        5.26, 5.25, 5.02, 4.50,
        3.18, 2.00, 1.94, 0.51,
        0.18, 0.18, 0.16, 0.12,
        0.13, 0.18, 0.19, 0.18,
        0.14, 0.09, 0.08, 0.07,
        0.08, 0.15, 0.14, 0.16,
        0.14, 0.12, 0.08, 0.09,
        0.07, 0.09, 0.09, 0.12,
        0.11, 0.13, 0.14, 0.24,
        0.36, 0.37, 0.40, 0.54,
        0.79, 0.95, 1.15, 1.20,
        1.45, 1.82, 1.95, 2.27
    ],
    # QoQ growth (%) of CSUSHPISA (Case-Shiller US National HPI, seasonally
    # adjusted), computed from the real FRED history over the same window.
    "HPI_growth": [
        0.42, -1.52, -1.85, -1.62,
        -2.16, -2.86, -2.83, -3.66,
        -3.83, -1.43, 0.22, -0.18,
        -1.13, -0.09, -1.50, -1.20,
        -1.15, -0.49, -0.49, -1.43,
        -0.18, 2.26, 1.55, 1.60,
        2.51, 3.15, 2.67, 2.02,
        1.50, 0.73, 0.94, 1.28,
        1.10, 1.01, 1.22, 1.57,
        1.14, 0.97, 1.35, 1.59,
        1.42, 1.20, 1.59, 1.74,
        1.64, 1.13, 1.10, 0.99
    ]
}


def _fetch_fred_series(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
    """Fetch one series from the official FRED REST API as a date-indexed Series."""
    import requests  # noqa: PLC0415

    resp = requests.get(
        f"{FRED_API_BASE_URL}/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
        },
        timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json()["observations"]
    s = pd.DataFrame(obs)[["date", "value"]]
    s["date"] = pd.to_datetime(s["date"])
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    return s.set_index("date")["value"].rename(series_id)


def download_or_generate_macro(
    output_path: Path, start: str = "2006-10-01", end: str = "2018-12-31"
) -> pd.DataFrame:
    """Downloads live macro data from the official FRED API, or falls back to
    the offline history if FRED_API_KEY is missing or the request fails.

    ``start`` defaults one quarter early so the first QoQ growth value
    (2007Q1) has a prior-quarter base to compute against.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FRED_API_KEY not set. Add it to .env (see .env.example) to pull live FRED data."
            )

        logger.info("Fetching macro data from the official FRED API...")
        series = {
            dest: _fetch_fred_series(fred_id, api_key, start, end)
            for fred_id, dest in _FRED_SERIES.items()
        }
        raw = pd.concat(series.values(), axis=1)
        raw.columns = list(series.keys())

        macro_q = raw.resample("QE").mean()
        macro_q["GDP_growth"] = macro_q["GDP"].pct_change() * 100
        macro_q["CPI_inflation"] = macro_q["CPIAUCSL"].pct_change() * 100
        macro_q["HPI_growth"] = macro_q["HPI"].pct_change() * 100

        macro_q = macro_q.dropna(
            subset=["UNRATE", "GDP_growth", "CPI_inflation", "FEDFUNDS", "HPI_growth"]
        ).reset_index()
        macro_q["quarter"] = macro_q["date"].dt.to_period("Q").astype(str)

        df = macro_q[
            ["quarter", "UNRATE", "GDP_growth", "CPI_inflation", "FEDFUNDS", "HPI_growth"]
        ]
        df.to_csv(output_path, index=False)
        logger.info(
            "Downloaded %d quarters of live FRED macro data to %s", len(df), output_path
        )
        return df

    except Exception as exc:
        logger.warning(
            "Failed to download from FRED (%s). Falling back to offline real historical macro dataset.",
            exc,
        )
        df = pd.DataFrame(_REAL_HISTORY)
        df.to_csv(output_path, index=False)
        logger.info("Saved offline historical macro dataset to %s", output_path)
        return df


if __name__ == "__main__":
    import logging as _logging
    from credit_risk.utils.logging import setup_logging
    setup_logging(_logging.INFO)
    
    out_dir = Path("data/processed")
    download_or_generate_macro(out_dir / "macro_quarterly.csv")
