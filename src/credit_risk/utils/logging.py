"""Centralised logging configuration for the credit_risk package."""

import sys, logging
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with UTF-8 safe stream handler."""
    stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.basicConfig(handlers=[handler], level=level, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; call setup_logging() first."""
    return logging.getLogger(name)
