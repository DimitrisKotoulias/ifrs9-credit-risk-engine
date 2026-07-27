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
import re
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


def _compact_count(value: object) -> str:
    """Human-scale loan count, e.g. 1,369,566 -> 1.37M."""
    n = _num(value)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{int(n):,}"


def update_headline_counts(readme: str, metrics: dict) -> str:
    """Rewrite the loan-count claims that sit outside the metrics table.

    Three distinct populations exist and the README used to conflate them: it labelled the
    resolved-outcome count as "accepted" and the modelling population as "resolved", so
    every figure was attached to the wrong name and the README contradicted the report's
    own abstract (docs/AUDIT.md finding A5; Flaws.md finding N19).

      accepted   = rows in the source file            (n_accepted_file)
      resolved   = rows with a good/bad outcome       (n_resolved_outcome)
      modelling  = train + test + OOT, net of the 2015 grey zone
    """
    accepted = _compact_count(metrics.get("n_accepted_file"))
    resolved = _compact_count(metrics.get("n_resolved_outcome"))
    modelling = _compact_count(
        _num(metrics.get("n_train")) + _num(metrics.get("n_test")) + _num(metrics.get("n_oot"))
    )
    readme = re.sub(
        r"(3-stage IFRS 9 ECL \u2014 on )[\d.,]+[MkK]?( Lending Club loans)",
        lambda m: f"{m.group(1)}{accepted}{m.group(2)}",
        readme,
    )
    readme = re.sub(
        r"(Latest full real-data run \(Lending Club 2007\u20132018, )[^)]*\)",
        lambda m: (
            f"{m.group(1)}~{accepted} accepted loans; {resolved} with a resolved "
            f"good/bad outcome; {modelling} in the modelling population after the 2015 "
            "grey-zone embargo)"
        ),
        readme,
    )
    return readme


def update_figure_captions(readme: str, metrics: dict) -> str:
    """Rewrite the numbers hand-typed into the Key Visualizations captions.

    These sat outside the METRICS block and had drifted several runs behind the table
    directly above them (Flaws.md finding N19).
    """
    gini_test = _num(metrics.get("gini"))
    gini_oot = _num(metrics.get("gini_oot"))
    if gini_test == gini_test and gini_oot == gini_oot:
        readme = re.sub(
            r"(\*\*ROC \u2014 in-time vs out-of-time\*\* \(Gini )[\d.]+ \u2192 [\d.]+\)",
            lambda m: f"{m.group(1)}{gini_test:.3f} \u2192 {gini_oot:.3f})",
            readme,
        )

    ec = metrics.get("econ_cap") or {}
    if ec.get("var") is not None and ec.get("es") is not None:
        readme = re.sub(
            r"(\*\*Portfolio loss distribution\*\* \u2014 VaR 99\.9% )\$[\d.]+bn, ES \$[\d.]+bn",
            lambda m: f"{m.group(1)}{_money(ec['var'])}, ES {_money(ec['es'])}",
            readme,
        )

    if metrics.get("total_ecl") is not None:
        readme = re.sub(
            r"(\u00b12 Z-factor vs )\$[\d.]+bn( baseline)",
            lambda m: f"{m.group(1)}{_money(metrics['total_ecl'])}{m.group(2)}",
            readme,
        )
    return readme


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
    new_readme = update_headline_counts(new_readme, metrics)
    new_readme = update_figure_captions(new_readme, metrics)
    README_PATH.write_text(new_readme, encoding="utf-8")
    print(f"README.md Key Results table regenerated from {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

