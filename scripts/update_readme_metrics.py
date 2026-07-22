"""Regenerate the README "Key Results" table from outputs/metrics.json.

Usage:
    python scripts/update_readme_metrics.py
    make readme

Rewrites the Markdown table between the ``<!-- METRICS:START -->`` /
``<!-- METRICS:END -->`` markers in README.md so the claim "the table is generated
from outputs/metrics.json" is literally true, rather than a hand-typed snapshot that
can silently drift from the last real pipeline run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "outputs" / "metrics.json"
README_PATH = ROOT / "README.md"

START_MARKER = "<!-- METRICS:START -->"
END_MARKER = "<!-- METRICS:END -->"


def _num(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _pct(value: object, precision: int = 1) -> str:
    v = _num(value)
    if v != v:  # NaN
        return "n/a"
    return f"{v * 100:.{precision}f}%"


def _money(value: object) -> str:
    """Format a dollar amount, abbreviating to bn/m for readability in the README."""
    v = _num(value)
    if v != v:
        return "n/a"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}bn"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}m"
    return f"${v:,.0f}"


def build_table(metrics: dict) -> str:
    """Build the Key Results markdown table from a metrics.json dict."""
    cutoff = metrics.get("cutoff_optimal_profit") or {}
    rwa_density = metrics.get("rwa_density", "n/a")
    if isinstance(rwa_density, (int, float)):
        rwa_density = _pct(rwa_density)

    auc_oot = _num(metrics.get("auc_oot"))
    gini_oot = _num(metrics.get("gini_oot"))
    ks_oot = _num(metrics.get("ks_oot"))
    psi = _num(metrics.get("psi_total"))
    mean_lgd = _num(metrics.get("mean_lgd"))
    downturn_lgd = _num(metrics.get("downturn_lgd"))

    cutoff_score = cutoff.get("cutoff", metrics.get("optimal_cutoff_threshold", 0))
    cutoff_approval = cutoff.get("approval_rate", metrics.get("optimal_approval_rate", 0.0))
    cutoff_bad = cutoff.get("bad_rate", metrics.get("optimal_bad_rate", 0.0))
    cutoff_raroc = cutoff.get("raroc", 0.0)

    rows = [
        ("PD AUC (OOT)", f"{auc_oot:.3f}"),
        ("Gini (OOT)", f"{gini_oot:.3f}"),
        ("KS (OOT)", f"{ks_oot:.3f}"),
        ("PSI (train → OOT)", f"{psi:.3f}"),
        ("Mean LGD (OOS-selected model)", f"{mean_lgd:.3f}"),
        ("Downturn LGD (p90)", f"{downturn_lgd:.3f}"),
        ("Portfolio EL", _money(metrics.get("total_el"))),
        ("Total RWA (IRB)", _money(metrics.get("total_rwa"))),
        ("RWA density", rwa_density),
        ("Total IFRS 9 ECL", _money(metrics.get("total_ecl"))),
        ("ECL coverage", _pct(metrics.get("ecl_coverage"))),
        (
            "Stage 2 / Stage 3 share",
            f"{_pct(metrics.get('stage2_pct'))} / {_pct(metrics.get('stage3_pct'))}",
        ),
        (
            "Operating cut-off",
            (
                f"score {int(_num(cutoff_score))} ({_pct(cutoff_approval)} approval, "
                f"{_pct(cutoff_bad)} bad rate, RAROC {_pct(cutoff_raroc)})"
            ),
        ),
    ]
    lines = ["| Metric | Value |", "|--------|-------|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(lines)


def main() -> int:
    if not METRICS_PATH.exists():
        print(
            f"error: {METRICS_PATH} not found -- run `make pipeline` first.",
            file=sys.stderr,
        )
        return 1
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    table = build_table(metrics)

    readme = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in readme or END_MARKER not in readme:
        print(
            f"error: {START_MARKER}/{END_MARKER} markers not found in README.md",
            file=sys.stderr,
        )
        return 1
    pre, rest = readme.split(START_MARKER, 1)
    _, post = rest.split(END_MARKER, 1)
    new_readme = f"{pre}{START_MARKER}\n{table}\n{END_MARKER}{post}"
    README_PATH.write_text(new_readme, encoding="utf-8")
    print(f"README.md Key Results table regenerated from {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

