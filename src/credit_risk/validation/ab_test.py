"""Paired bootstrap A/B test for champion vs challenger discrimination.

Point-estimate AUC/Gini comparisons cannot say whether a challenger's edge is real or
noise. A paired bootstrap resamples the same held-out rows for both models and builds a
confidence interval on the *difference* in Gini. If that interval excludes zero the
improvement is statistically significant. This complements the analytic DeLong test
(``validation/discrimination.delong_test``) already reported.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def paired_bootstrap_gini(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    """Paired bootstrap CIs for Gini of A, B and the difference (B - A).

    Parameters
    ----------
    y_true:
        Binary outcomes (1 = default).
    pred_a, pred_b:
        Champion (A) and challenger (B) predicted probabilities on the same rows.
    n_boot:
        Number of bootstrap resamples.
    ci:
        Central confidence level (e.g. 0.95 → 2.5/97.5 percentiles).
    seed:
        RNG seed.

    Returns
    -------
    dict with ``gini_a, gini_b, diff`` (each ``{median, lo, hi}``), ``significant``
    (bool: does the difference CI exclude zero?), ``ci`` and ``n_boot_valid``.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    n = len(y)
    if not (n == len(a) == len(b)):
        raise ValueError("y_true, pred_a and pred_b must have equal length")

    rng = np.random.default_rng(seed)
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0

    ginis_a: list[float] = []
    ginis_b: list[float] = []
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.min() == yi.max():  # need both classes for AUC
            continue
        ga = 2.0 * roc_auc_score(yi, a[idx]) - 1.0
        gb = 2.0 * roc_auc_score(yi, b[idx]) - 1.0
        ginis_a.append(ga)
        ginis_b.append(gb)
        diffs.append(gb - ga)

    if not diffs:
        nan3 = {"median": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"gini_a": nan3, "gini_b": dict(nan3), "diff": dict(nan3),
                "significant": False, "ci": ci, "n_boot_valid": 0}

    def _pct(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals)
        return {
            "median": float(np.percentile(arr, 50)),
            "lo": float(np.percentile(arr, lo_q)),
            "hi": float(np.percentile(arr, hi_q)),
        }

    diff_stats = _pct(diffs)
    significant = bool(diff_stats["lo"] > 0.0 or diff_stats["hi"] < 0.0)
    result = {
        "gini_a": _pct(ginis_a),
        "gini_b": _pct(ginis_b),
        "diff": diff_stats,
        "significant": significant,
        "ci": ci,
        "n_boot_valid": len(diffs),
    }
    logger.info(
        "Paired bootstrap Gini: A=%.4f B=%.4f diff=%.4f [%.4f, %.4f] significant=%s",
        result["gini_a"]["median"], result["gini_b"]["median"],
        diff_stats["median"], diff_stats["lo"], diff_stats["hi"], significant,
    )
    return result
