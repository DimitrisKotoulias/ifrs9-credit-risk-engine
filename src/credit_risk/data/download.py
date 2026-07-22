"""Download Lending Club data from Kaggle.

Usage:
    python -m credit_risk.data.download

Requires:
    %USERPROFILE%\\.kaggle\\kaggle.json  (Windows)
    ~/.kaggle/kaggle.json               (Linux/macOS)

Download your Kaggle API token from:
    https://www.kaggle.com/settings  →  Account  →  Create New Token
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from credit_risk.utils.config import load_config

logger = logging.getLogger(__name__)

_KAGGLE_JSON_WINDOWS = Path.home() / ".kaggle" / "kaggle.json"
_KAGGLE_SETUP_URL = "https://www.kaggle.com/settings"


def _check_kaggle_credentials() -> None:
    """Raise informative error if kaggle.json is missing."""
    if not _KAGGLE_JSON_WINDOWS.exists():
        raise FileNotFoundError(
            "\n"
            "Kaggle credentials not found.\n"
            f"Expected: {_KAGGLE_JSON_WINDOWS}\n\n"
            "Steps to fix:\n"
            f"  1. Go to {_KAGGLE_SETUP_URL}\n"
            "  2. Scroll to 'API' section → click 'Create New Token'\n"
            "  3. Save the downloaded kaggle.json to:\n"
            f"     {_KAGGLE_JSON_WINDOWS}\n"
            "  4. Re-run: make data-download\n"
        )


def download_lending_club(force: bool = False) -> Path:
    """Download Lending Club CSVs via Kaggle API.

    Parameters
    ----------
    force:
        If True, re-download even if files already present.

    Returns
    -------
    Path
        Path to data/raw/ directory.
    """
    cfg = load_config()
    raw_dir = Path(cfg.data.raw_dir)
    accepted_path = raw_dir / cfg.data.accepted_file
    rejected_path = raw_dir / cfg.data.rejected_file

    if not force and accepted_path.exists() and rejected_path.exists():
        logger.info(
            "Lending Club data already present in %s — skipping download.", raw_dir
        )
        return raw_dir

    _check_kaggle_credentials()

    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset '%s' → %s", cfg.data.kaggle_dataset, raw_dir)

    # Import here so import errors surface only when actually downloading
    try:
        import kaggle  # noqa: PLC0415
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            cfg.data.kaggle_dataset,
            path=str(raw_dir),
            unzip=True,
            quiet=False,
        )
    except ImportError as exc:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle"
        ) from exc

    # Verify files landed
    missing = [p for p in (accepted_path, rejected_path) if not p.exists()]
    if missing:
        # Kaggle sometimes uses slightly different filenames — list what's there
        present = list(raw_dir.iterdir())
        raise FileNotFoundError(
            f"Expected files not found after download: {missing}\n"
            f"Files present in {raw_dir}: {present}\n"
            "Check the dataset name in config/config.yaml."
        )

    logger.info("Download complete. Files:\n  %s\n  %s", accepted_path, rejected_path)
    return raw_dir


if __name__ == "__main__":
    import logging as _logging

    from credit_risk.utils.logging import setup_logging

    setup_logging(_logging.INFO)
    download_lending_club()
