"""Pre-build QA consistency checks for the model risk report (Fix 3.3 + 1.1#4).

Run automatically by render_latex.py before PDF compilation. Verifies
cross-table numerical identities in outputs/metrics.json and guards the
generated .tex against unreplaced template variables.

Raises QAError on any failure so the build stops instead of shipping an
internally inconsistent report.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REL_TOL = 0.001  # 0.1% relative tolerance per upgrades.md Fix 3.3


class QAError(AssertionError):
    """A report consistency check failed."""


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def check_rwa_density(metrics: dict, failures: list[str]) -> None:
    """RWA_IRB / EAD_total must equal the reported RWA density."""
    rwa = metrics.get("total_rwa")
    ead = metrics.get("total_ead_portfolio")
    density_str = str(metrics.get("rwa_density", "")).replace("%", "").strip()
    if not (rwa and ead and density_str):
        return
    density_reported = float(density_str) / 100.0
    density_computed = rwa / ead
    _check(
        abs(density_computed - density_reported) < 0.001,
        f"RWA density mismatch: computed {density_computed:.4f} vs reported {density_reported:.4f}",
        failures,
    )


def check_ecl_coverage(metrics: dict, failures: list[str]) -> None:
    """ECL_total / EAD_total must equal the reported coverage ratio."""
    ecl = metrics.get("total_ecl")
    ead = metrics.get("total_ead_portfolio")
    cov = metrics.get("ecl_coverage")
    if not (ecl and ead and cov):
        return
    _check(
        abs(ecl / ead - cov) < 1e-4,
        f"ECL coverage mismatch: computed {ecl / ead:.6f} vs reported {cov:.6f}",
        failures,
    )


def check_capital_identity(metrics: dict, failures: list[str]) -> None:
    """Minimum capital (8% of RWA) must equal RWA / 12.5."""
    rwa = metrics.get("total_rwa")
    if not rwa:
        return
    _check(
        _rel_err(rwa * 0.08, rwa / 12.5) < REL_TOL,
        "Capital identity broken: RWA*0.08 != RWA/12.5",
        failures,
    )
    stress_rwa = metrics.get("stress_rwa")
    stress_cap = metrics.get("stress_capital_req")
    if stress_rwa and stress_cap:
        _check(
            _rel_err(stress_cap, stress_rwa * 0.08) < REL_TOL,
            f"Stress capital mismatch: reported {stress_cap:,.0f} vs RWA*8% {stress_rwa * 0.08:,.0f}",
            failures,
        )


def check_cutoff_optimum(metrics: dict, failures: list[str]) -> None:
    """The cited operating cutoff must be a traceable grid row and equal the
    risk-appetite cutoff (profit max subject to the approved bad-rate ceiling)."""
    grid = metrics.get("cutoff_strategy_table") or []
    opt = metrics.get("cutoff_optimal_profit") or {}
    if not (grid and opt):
        return
    # (a) the cited optimum must correspond to an actual grid row (traceability)
    grid_row = next((r for r in grid if r.get("cutoff") == opt.get("cutoff")), None)
    _check(
        grid_row is not None,
        f"Cited operating cutoff {opt.get('cutoff')} is not a row in the swept grid",
        failures,
    )
    if grid_row is not None:
        for key in ("approval_rate", "bad_rate", "expected_profit", "raroc"):
            if key in opt and key in grid_row:
                _check(
                    _rel_err(grid_row[key], opt[key]) < REL_TOL,
                    f"Operating-cutoff {key} mismatch: grid {grid_row[key]} vs cited {opt[key]}",
                    failures,
                )
    # (b) it must equal the risk-appetite cutoff, and honour the bad-rate ceiling
    max_bad = metrics.get("cutoff_max_bad_rate")
    if max_bad is not None:
        _check(
            opt.get("bad_rate", 1.0) <= max_bad + 1e-9,
            f"Operating cutoff bad rate {opt.get('bad_rate')} exceeds appetite ceiling {max_bad}",
            failures,
        )
        try:
            from credit_risk.business.cutoff import risk_appetite_cutoff  # noqa: PLC0415
        except Exception:  # pragma: no cover - import guard
            return
        expected = risk_appetite_cutoff(grid, max_bad_rate=max_bad)
        if expected is not None:
            _check(
                expected["cutoff"] == opt.get("cutoff"),
                f"Cited cutoff {opt.get('cutoff')} != risk-appetite cutoff {expected['cutoff']} (ceiling {max_bad})",
                failures,
            )


def check_scenario_dr(metrics: dict, failures: list[str]) -> None:
    """Implied default rate per scenario must equal beta·x from the projection
    coefficients (sign-adjusted where those were used)."""
    # Scenario projections use the sign-adjusted coefficients when present.
    elas = metrics.get("macro_elasticities_adjusted") or metrics.get("macro_elasticities") or {}
    inputs = metrics.get("macro_scenario_inputs") or {}
    preds = metrics.get("macro_predictions") or {}  # stored in percent
    if not (elas and inputs and preds):
        return
    for scen, x in inputs.items():
        if scen not in preds:
            continue
        dr = elas.get("const", 0.0) + sum(
            elas.get(k, 0.0) * v for k, v in x.items()
        )
        dr = min(max(dr, 1e-4), 0.99)
        reported = preds[scen] / 100.0
        _check(
            _rel_err(dr, reported) < REL_TOL,
            f"Scenario '{scen}' implied DR mismatch: recomputed {dr:.4%} vs reported {reported:.4%}",
            failures,
        )


def check_scenario_ordering(metrics: dict, failures: list[str]) -> None:
    """Implied default rates must follow Downside >= Baseline >= Upside."""
    preds = metrics.get("macro_predictions") or {}
    up, base, down = preds.get("upside"), preds.get("baseline"), preds.get("downside")
    if None in (up, base, down):
        return
    _check(
        down >= base >= up,
        f"Scenario DR ordering violated: downside {down:.3f} / baseline {base:.3f} / upside {up:.3f} "
        "(expected downside >= baseline >= upside)",
        failures,
    )


def check_irb_sa_direction(metrics: dict, failures: list[str]) -> None:
    """IRB vs SA RWA direction must be consistent with the reported RWA density
    (density above the flat 75% SA weight <=> IRB RWA above SA RWA = surcharge)."""
    rwa = metrics.get("total_rwa")
    rwa_sa = metrics.get("total_rwa_sa")
    density_str = str(metrics.get("rwa_density", "")).replace("%", "").strip()
    if not (rwa and rwa_sa and density_str):
        return
    density = float(density_str) / 100.0
    _check(
        (rwa > rwa_sa) == (density > 0.75),
        f"IRB/SA direction inconsistent: total_rwa {rwa:,.0f} vs SA {rwa_sa:,.0f} "
        f"but RWA density {density:.3f} vs 0.75 SA flat weight",
        failures,
    )


def check_vintage_pd_ratio(metrics: dict, failures: list[str]) -> None:
    """Spot-check: predicted PD / actual DR must equal the reported PD Ratio."""
    rows = metrics.get("pd_backtest_vintage") or []
    for row in rows[:5] + rows[-5:]:
        pred = row.get("predicted_pd") or row.get("pred_pd")
        actual = row.get("actual_dr") or row.get("actual_default_rate")
        ratio = row.get("pd_ratio") or row.get("ratio")
        if pred is None or actual is None or ratio is None or not actual:
            continue
        _check(
            _rel_err(pred / actual, ratio) < 0.01,
            f"Vintage {row.get('vintage')}: PD ratio {ratio} != predicted/actual {pred / actual:.4f}",
            failures,
        )


def check_no_unreplaced_vars(tex_path: Path, failures: list[str]) -> None:
    """Fix 1.1#4: generated .tex must contain no unreplaced VAR_ or __TOKEN__ placeholders.

    Also catches LaTeX-escaped variants (e.g. ``VAR\\_N\\_OOT``) which evade a plain
    ``VAR_`` search and a naive ``str.replace("VAR_N_OOT", ...)`` — this was the actual
    source of the placeholders that reached the shipped PDF.
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    # VAR followed by (optionally backslash-escaped) underscore segments.
    hits = sorted(set(re.findall(r"VAR(?:\\?_)[A-Z0-9]+(?:\\?_[A-Z0-9]+)*", text)))
    _check(not hits, f"Unreplaced template variables in {tex_path.name}: {hits}", failures)
    tokens = sorted(set(re.findall(r"__[A-Z0-9_]{3,}__", text)))
    _check(not tokens, f"Unreplaced __TOKEN__ placeholders in {tex_path.name}: {tokens}", failures)


def check_es_ge_var(metrics: dict, failures: list[str]) -> None:
    """Economic-capital tail measures must obey ES >= VaR >= EL >= 0."""
    ec = metrics.get("econ_cap")
    if not ec:
        return
    el = float(ec.get("expected_loss", 0.0))
    var = float(ec.get("var", 0.0))
    es = float(ec.get("es", 0.0))
    _check(
        es >= var >= el >= 0.0,
        f"Economic-capital ordering violated: EL={el:.0f}, VaR={var:.0f}, ES={es:.0f} "
        "(require ES >= VaR >= EL >= 0)",
        failures,
    )


def check_benchmarks_sourced(failures: list[str]) -> None:
    """Every literature benchmark must be registry-backed with a resolvable citation.

    Guards against re-introducing a hand-typed/fabricated benchmark row (the original
    LGD R^2 defect): each Benchmark in reports/benchmarks.py must carry a non-empty
    source citation + locator, a well-ordered range, and a bibkey that actually exists
    in the report's .bib so the citation resolves.
    """
    try:
        from benchmarks import BENCHMARKS  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import guard
        _check(False, f"Cannot import benchmark registry: {exc}", failures)
        return

    bib_path = Path(__file__).resolve().parent / "model_risk_report.bib"
    bib_keys: set[str] = set()
    if bib_path.exists():
        bib_text = bib_path.read_text(encoding="utf-8", errors="replace")
        bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))

    for key, b in BENCHMARKS.items():
        _check(bool(b.source_bibkey), f"Benchmark '{key}' has no source_bibkey", failures)
        _check(bool(b.source_locator), f"Benchmark '{key}' has no source_locator", failures)
        _check(b.low <= b.high, f"Benchmark '{key}' range is inverted ({b.low} > {b.high})", failures)
        _check(bool(b.metric_key), f"Benchmark '{key}' has no metric_key", failures)
        if bib_keys:
            _check(
                b.source_bibkey in bib_keys,
                f"Benchmark '{key}' cites '{b.source_bibkey}' which is absent from the .bib",
                failures,
            )


def check_no_fabricated_benchmark(tex_path: Path, failures: list[str]) -> None:
    """The generated .tex must not contain the old fabricated static LGD R^2 band.

    The LGD R^2 row is now driven by the live computed metric; a literal ``0.09 - 0.15``
    (or ``0.09 -- 0.15``) reappearing means a hand-typed value was smuggled back in.
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    _check(
        not re.search(r"0\.09\s*-{1,2}\s*0\.15", text),
        f"Fabricated static LGD R^2 band '0.09-0.15' found in {tex_path.name}",
        failures,
    )


def check_citations_resolve(tex_path: Path, failures: list[str]) -> None:
    """Every \\parencite/\\textcite/\\cite key in the .tex must exist in the .bib.

    Guards against citation drift such as citing a work that was never added to the
    bibliography (the SHAP/Lundberg and Bellotti-2009 survival defects), or leaving a
    dangling key after a citation is corrected.
    """
    bib_path = Path(__file__).resolve().parent / "model_risk_report.bib"
    if not bib_path.exists():
        return
    bib_keys = set(
        re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_path.read_text(encoding="utf-8", errors="replace"))
    )
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    cited: set[str] = set()
    for m in re.finditer(
        r"\\(?:parencite|textcite|autocite|footcite|cite)\s*(?:\[[^\]]*\])*\{([^}]*)\}", text
    ):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                cited.add(key)
    missing = sorted(cited - bib_keys)
    _check(not missing, f"Citations with no matching .bib entry: {missing}", failures)


def check_lgd_r2_consistency(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The rejected two-stage LGD R^2 quoted in prose must track the live metric.

    The value is now substituted from ``lgd_model_comparison.champion.r2`` in both the
    §7.7 body and the benchmark comment; a reappearance of the old hand-typed ``-1.13``
    (or absence of the live value) means a stale literal was smuggled back in.
    """
    ts = ((metrics.get("lgd_model_comparison") or {}).get("champion") or {}).get("r2")
    if ts is None:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    expected = f"{float(ts):.2f}"
    _check(
        "-1.13" not in text,
        "Stale two-stage LGD $R^2$ '-1.13' found in report (must track lgd_model_comparison.champion.r2)",
        failures,
    )
    _check(
        expected in text,
        f"Two-stage LGD $R^2$ {expected} (from metric) not found in report body",
        failures,
    )


def check_recalibration_claim(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The report must not claim recalibration 'materially improves' slope/intercept when
    the metrics show both moved away from their targets on OOT."""
    comp = metrics.get("calibration_comparison") or {}
    before, after = comp.get("before") or {}, comp.get("after") or {}
    if not (before and after):
        return
    slope_worse = abs(after.get("slope", 0.0) - 1.0) >= abs(before.get("slope", 0.0) - 1.0)
    intercept_worse = abs(after.get("intercept", 0.0)) >= abs(before.get("intercept", 0.0))
    if slope_worse and intercept_worse:
        text = tex_path.read_text(encoding="utf-8", errors="replace")
        _check(
            "materially improves the calibration slope" not in text,
            "Report claims recalibration 'materially improves the calibration slope' but the "
            "metrics show slope and intercept both moved away from target",
            failures,
        )


def check_recalibration_applied(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The report must not describe a recalibration that was never attached.

    The calibrator is fitted only when the in-time test Hosmer-Lemeshow test rejects. When
    it does not, ``predict_proba`` returns raw PDs and the before/after table is a
    diagnostic. The shipped report previously claimed the transform was "applied to the OOT
    set", "retained", and feeding EL/RWA/staging (AUDIT-A1).
    """
    cal = metrics.get("calibration") or {}
    if cal.get("isotonic_applied") is None and "recalibration_applied" not in cal:
        return
    comp = metrics.get("calibration_comparison") or {}
    applied = bool(
        cal.get("recalibration_applied",
                comp.get("applied_in_production", cal.get("isotonic_applied", False)))
    )
    # The gate's own record must agree with the attach state.
    gate = cal.get("recalibration_gate") or {}
    if gate:
        _check(
            bool(gate.get("accepted", False)) == applied,
            f"recalibration_gate.accepted={gate.get('accepted')} contradicts "
            f"recalibration_applied={applied}",
            failures,
        )
        _check(
            applied or cal.get("method_chosen", "none") == "none",
            f"method_chosen={cal.get('method_chosen')!r} while no transform is attached",
            failures,
        )
    if applied:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    banned = [
        "and applied to the OOT set",
        "we retain the recalibrated PDs",
        "corrected via the isotonic regression recalibration",
    ]
    for phrase in banned:
        _check(
            phrase not in text,
            f"Report claims recalibration is deployed ({phrase!r}) but no calibrator is "
            "attached to the production scorecard (calibration.isotonic_applied is false)",
            failures,
        )
    _check(
        cal.get("method_chosen") != "isotonic" or cal.get("isotonic_applied", False),
        "metrics.json reports method_chosen='isotonic' while isotonic_applied is false; "
        "record 'none' when nothing is attached",
        failures,
    )


def check_cutoff_corner_direction(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """"Approve everyone" wording is only valid if the unconstrained argmax really is
    full approval. On a book where every grid RAROC is negative the argmax is the most
    *exclusive* non-empty cutoff (AUDIT-A2)."""
    corner = metrics.get("cutoff_raroc_max") or metrics.get("cutoff_profit_argmax") or {}
    if not corner:
        return
    approval = float(corner.get("approval_rate", 0.0))
    if approval >= 0.99:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    banned = [
        "approving the entire through-the-door population",
        "implies 100\\% approval",
        "(full approval)",
    ]
    for phrase in banned:
        _check(
            phrase not in text,
            f"Report describes the unconstrained cutoff corner as full approval ({phrase!r}) "
            f"but the argmax row approves only {approval:.4%} of the population",
            failures,
        )


def check_capital_charge_rate(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The charge netted out of expected profit is the cost of capital, not the RAROC
    hurdle (AUDIT-A3)."""
    coc = metrics.get("cutoff_cost_of_capital")
    hurdle = metrics.get("cutoff_raroc_hurdle")
    if coc is None or hurdle is None or abs(float(coc) - float(hurdle)) < 1e-9:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    hurdle_pct = f"{float(hurdle) * 100:.2f}\\%"
    _check(
        f"a {hurdle_pct} charge on economic capital" not in text,
        f"Report states a {hurdle_pct} charge on economic capital, but the cost of capital "
        f"actually charged is {float(coc) * 100:.2f}%",
        failures,
    )


def check_population_counts(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """Every large loan count in the prose must trace to a metrics.json population.

    The abstract previously carried these as hard-coded literals, and the pipeline exposed
    only a mis-named `n_accepted_raw` (actually the resolved-outcome count), so a drifted
    figure could not be detected (AUDIT-A5).
    """
    n_train = int(metrics.get("n_train", 0))
    n_test = int(metrics.get("n_test", 0))
    n_oot = int(metrics.get("n_oot", 0))
    n_resolved = int(metrics.get("n_resolved_outcome", metrics.get("n_accepted_raw", 0)))
    n_modelling = n_train + n_test + n_oot
    known = {
        int(metrics.get("n_accepted_file", 0)),
        n_resolved,
        int(metrics.get("n_rejected_raw", 0)),
        n_train,
        n_test,
        n_oot,
        n_modelling,
        max(0, n_resolved - n_modelling),  # grey-zone loans dropped by the split
    }
    known.discard(0)
    if not known:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    # Comma-grouped integers with 2+ groups: portfolio-scale populations. Currency figures
    # in this report are always prefixed with \$ (optionally followed by a minus sign), so
    # they are stripped out first rather than matched around.
    text_no_currency = re.sub(r"\\?\$-?\s?[\d,]+(?:\.\d+)?", " ", text)
    for m in re.finditer(r"\b(\d{1,3}(?:,\d{3}){2,})\b", text_no_currency):
        value = int(m.group(1).replace(",", ""))
        # Only police the population range; monetary totals are billions and are checked
        # by the identity tests instead.
        if not (100_000 <= value <= 50_000_000):
            continue
        _check(
            value in known,
            f"Population count {m.group(1)} in the report does not match any metrics.json "
            f"population {sorted(known)}",
            failures,
        )


def check_challenger_feature_parity(metrics: dict, failures: list[str]) -> None:
    """Champion and challenger must consume the same number of predictors.

    ``PDChallenger`` used to silently drop requested features missing from the raw frame,
    so the tree models trained on 13 of the scorecard's 15 predictors while the report
    claimed parity (AUDIT-A12 / B2).
    """
    shap_rows = (metrics.get("challenger") or {}).get("shap_mean_abs") or []
    if not shap_rows:
        return
    root = Path(__file__).resolve().parent.parent
    sc_path = root / "outputs" / "scorecard_tables.json"
    if not sc_path.exists():
        return
    with open(sc_path, encoding="utf-8") as f:
        selected = json.load(f).get("selected_features", [])
    if not selected:
        return
    _check(
        len(shap_rows) == len(selected),
        f"Challenger feature parity broken: challenger used {len(shap_rows)} features vs "
        f"the scorecard's {len(selected)}; the champion/challenger comparison is not "
        "like-for-like",
        failures,
    )



def check_scenario_axes_separate(metrics: dict, failures: list[str]) -> None:
    """Upside and downside must differ on every macro axis.

    FEDFUNDS previously floored to 0.1 in both scenarios and CPI_inflation carried no
    delta, so two of five axes contributed nothing to scenario separation while the
    report tabulated them as assumptions (AUDIT-C5).
    """
    inputs = metrics.get("macro_scenario_inputs") or {}
    up, down = inputs.get("upside") or {}, inputs.get("downside") or {}
    if not (up and down):
        return
    identical = [
        k for k in up
        if k in down and abs(float(up[k]) - float(down[k])) < 1e-9
    ]
    _check(
        not identical,
        f"Macro scenario axes identical in upside and downside: {identical}; "
        "they contribute nothing to scenario separation",
        failures,
    )



def check_no_unescaped_percent(tex_path: Path, failures: list[str]) -> None:
    """No inline unescaped ``%`` in the generated .tex.

    A percent sign is a LaTeX comment: it deletes the rest of the source line from the
    PDF with no error and no warning. Because the template writes each paragraph as one
    long line, a single stray ``%`` in a value injected from Python can silently remove
    hundreds of characters of evidence -- which is exactly what happened to the
    recalibration decision in an earlier build (AUDIT-A1).

    Whole-line comments (optionally indented) are legitimate and ignored.
    """
    failures_before = len(failures)
    for lineno, line in enumerate(tex_path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue  # a real comment line
        for col, ch in enumerate(line):
            if ch != "%":
                continue
            if col > 0 and line[col - 1] == "\\":
                continue  # escaped
            _check(
                False,
                f"Unescaped '%' at {tex_path.name}:{lineno}:{col + 1} truncates the rest "
                f"of the line in the PDF: ...{line[max(0, col - 60):col + 1]!r}",
                failures,
            )
            break
        if len(failures) - failures_before >= 5:
            failures.append("... further unescaped '%' occurrences suppressed")
            return


# ─────────────────────────────────────────────────────────────────────────────
# Round 3 guards (the internal review log).
#
# Every guard above tests a numeric identity inside metrics.json or bans a fixed
# phrase. None of them could catch the Round 3 findings, because those were cases
# where a *methodological sentence* did not describe what the code did. The guards
# below close that class: each one ties a claim in the report, or a headline number,
# back to the provenance the pipeline now records.
# ─────────────────────────────────────────────────────────────────────────────


def check_binner_is_optbinning(metrics: dict, failures: list[str]) -> None:
    """A silent fallback to the manual binner changes every number in the report.

    ``_try_optbinning`` used to swallow every exception, so an unrelated failure inside
    optbinning (e.g. a scikit-learn incompatibility) quietly substituted a completely
    different binner — different bins, different WoE, different surviving features — and
    the report still built (FLAWS-N32). Set ``allow_fallback_binner: true`` in
    metrics.json to override deliberately.
    """
    binner = metrics.get("binner")
    if binner is None:
        return
    if metrics.get("allow_fallback_binner"):
        return
    _check(
        binner == "optbinning",
        f"WoE bins were produced by {binner!r}, not optbinning. Every scorecard number in "
        "the report belongs to a different model than the documented one; set "
        "allow_fallback_binner=true in metrics.json to accept this deliberately",
        failures,
    )


def check_no_phase_failures(metrics: dict, failures: list[str]) -> None:
    """An optional phase that dropped out takes its table with it, silently.

    The xgboost import in pd_challenger.py is the worked example: on a clean install it
    raised, the whole challenger benchmark vanished, and only a non-fatal warning marked
    it (FLAWS-N14).
    """
    pf = metrics.get("phase_failures")
    if not pf:
        return
    messages = "; ".join(str(f.get("message", f)) for f in pf)
    _check(
        False,
        f"{len(pf)} enhancement phase(s) failed non-fatally, so their tables are missing "
        f"from the report: {messages}",
        failures,
    )


def check_recalibration_gate_chronology(metrics: dict, failures: list[str]) -> None:
    """A gate described as out-of-time must actually split on time.

    The gate fell back to row order whenever no ordering key was passed, which turned an
    out-of-TIME test into a random holdout while the report continued to describe it as
    chronological (FLAWS-N3).
    """
    gate = (metrics.get("calibration") or {}).get("recalibration_gate") or {}
    if not gate:
        return
    basis = gate.get("split_basis")
    _check(
        basis != "positional",
        "recalibration gate split the OOT window by ROW ORDER, not by date: it is not an "
        "out-of-time test and must not be described as one",
        failures,
    )
    fit_max, eval_min = gate.get("fit_slice_max_date"), gate.get("eval_slice_min_date")
    if fit_max and eval_min:
        _check(
            str(fit_max) <= str(eval_min),
            f"recalibration gate slices overlap in time: fit slice ends {fit_max}, eval "
            f"slice starts {eval_min}",
            failures,
        )


def check_calibration_table_direction(metrics: dict, failures: list[str]) -> None:
    """If the gate accepted a transform, the published before/after table must agree.

    The shipped table was built from a throwaway isotonic fitted on the wrong partition,
    so every row moved away from its target directly beside prose reporting an accepted
    improvement (FLAWS-N6).
    """
    gate = (metrics.get("calibration") or {}).get("recalibration_gate") or {}
    comp = metrics.get("calibration_comparison") or {}
    if not gate.get("accepted") or not comp:
        return
    before, after = comp.get("before") or {}, comp.get("after") or {}
    if not (before and after):
        return
    _check(
        comp.get("measured_on") == "gate_eval_slice",
        f"calibration_comparison was measured on {comp.get('measured_on')!r}, not on the "
        "slice the gate actually judged; the table and the acceptance decision can "
        "disagree",
        failures,
    )
    actual = float(after.get("actual_dr", before.get("actual_dr", 0.0)))

    # The gate's own acceptance criteria must hold on the published numbers, or the table
    # and the decision are describing different things.
    for name, key, target in (("aggregate PD ratio", "ratio", 1.0), ("Brier", "brier", 0.0)):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            continue
        _check(
            abs(float(a) - target) <= abs(float(b) - target) + 1e-9,
            f"gate accepted the recalibration on {name}, but the published table shows it "
            f"moving AWAY from target ({float(b):.4f} -> {float(a):.4f})",
            failures,
        )

    # Beyond those two, an isotonic fit legitimately trades a little slope for a large
    # gain in level, so individual metrics may regress. What must NOT happen is the
    # pre-fix picture: an accepted gate beside a table in which everything worsens.
    moved_toward, moved_away = [], []
    for name, key, target in (
        ("Brier", "brier", 0.0),
        ("calibration slope", "slope", 1.0),
        ("calibration intercept", "intercept", 0.0),
        ("expected default rate", "expected_dr", actual),
    ):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            continue
        (moved_toward if abs(float(a) - target) <= abs(float(b) - target) + 1e-9
         else moved_away).append(name)

    _check(
        not moved_away or len(moved_toward) > len(moved_away),
        "gate accepted the recalibration but the published table is not an improvement: "
        f"{len(moved_away)} of {len(moved_toward) + len(moved_away)} metrics moved away "
        f"from target ({', '.join(moved_away)})",
        failures,
    )


def check_pd_horizon_consistency(metrics: dict, failures: list[str]) -> None:
    """Basel IRB and the per-annum P&L both require a ONE-YEAR PD.

    The scorecard target is terminal loan status, i.e. a lifetime default flag. Feeding it
    straight into the IRB formula produced an RWA density of 228.8% and an RWA that
    *falls* under stress, because the IRB capital function is concave and was being
    evaluated far out on its flat region (FLAWS-N1).
    """
    horizon = metrics.get("pd_horizon_basel")
    mean_pd = metrics.get("mean_pd_basel", metrics.get("mean_pd_12m"))
    if horizon is None and mean_pd is None:
        return
    if horizon not in (None, "12m"):
        # An explicit non-12m horizon is a deliberate, disclosed choice.
        return
    if mean_pd is None:
        return
    _check(
        float(mean_pd) <= 0.15,
        f"mean PD entering the Basel IRB formula is {float(mean_pd):.4f}, far above the "
        "range a one-year consumer PD can occupy — a lifetime PD is being used where a "
        "12-month PD is required (set pd_horizon_basel explicitly to override)",
        failures,
    )
    # And the mirror-image failure. This check had only an upper bound, so it passed
    # cheerfully while the hazard model was returning a mean 12-month PD of ~1e-20 and the
    # Stage 1 provision was exactly zero: the guard written to protect the PD horizon was
    # blind in the direction the engine actually broke.
    _check(
        float(mean_pd) >= 0.002,
        f"mean PD entering the Basel IRB formula is {float(mean_pd):.3g} — a consumer book "
        "with a ~20% lifetime default rate cannot have a 12-month PD near zero; the PD "
        "series feeding Basel is degenerate",
        failures,
    )


def check_stress_direction(metrics: dict, failures: list[str]) -> None:
    """Stressed RWA below base RWA is a red flag, not a result.

    It happens when PD is already so high that the concave IRB capital function is past
    its peak, which is itself a symptom of the lifetime-PD problem above
    (FLAWS-N18).
    """
    base, stressed = metrics.get("total_rwa"), metrics.get("stress_rwa")
    if base is None or stressed is None:
        return
    _check(
        float(stressed) >= float(base),
        f"stressed RWA ({float(stressed):,.0f}) is BELOW base RWA ({float(base):,.0f}); "
        "the IRB capital function is being evaluated past its peak, which indicates the "
        "PD horizon feeding it is wrong",
        failures,
    )


def check_vintage_calibration_band(metrics: dict, failures: list[str]) -> None:
    """Vintage groups whose predicted/actual ratio is far off must not pass unremarked.

    Applying an OOT-era calibrator to the whole book leaves the development vintages
    over-predicting by ~50% (FLAWS-N5).
    """
    groups = metrics.get("vintage_calibration") or []
    offenders = []
    for row in groups:
        # pd_ratio_raw is the ratio for the PD the pipeline actually deploys; the column
        # name is historical (it predates the calibrator being attached in production).
        ratio = row.get("pd_ratio_raw", row.get("ratio"))
        if ratio is None:
            continue
        if not (0.80 <= float(ratio) <= 1.25):
            offenders.append(f"{row.get('group', '?')}={float(ratio):.3f}")
    _check(
        not offenders or bool(metrics.get("vintage_calibration_disclosed")),
        "vintage groups outside the [0.80, 1.25] predicted/actual band without disclosure: "
        + ", ".join(offenders)
        + " (set vintage_calibration_disclosed=true once the report explains the driver)",
        failures,
    )


def check_feature_selection_stages(metrics: dict, failures: list[str]) -> None:
    """The selection funnel must be monotone and complete.

    The report described two stages (IV, VIF) while the code ran four; the two undisclosed
    stages are what removed int_rate despite it carrying the highest IV in the table
    (FLAWS-N29).
    """
    stages = metrics.get("feature_selection_stages") or {}
    if not stages:
        return
    order = ["n_candidates", "n_after_iv", "n_after_vif", "n_after_elasticnet",
             "n_after_sign_check"]
    present = [(k, stages.get(k)) for k in order if stages.get(k) is not None]
    _check(
        len(present) == len(order),
        "feature_selection_stages is missing stage counts "
        f"({[k for k in order if stages.get(k) is None]}); the report cannot state the "
        "real funnel",
        failures,
    )
    for (k_prev, v_prev), (k_next, v_next) in zip(present, present[1:]):
        _check(
            int(v_next) <= int(v_prev),
            f"feature selection funnel is not monotone: {k_next}={v_next} exceeds "
            f"{k_prev}={v_prev}",
            failures,
        )


def check_no_orphan_figures(failures: list[str], figures_root: Path | None = None,
                            tex_path: Path | None = None) -> None:
    """Every PNG under reports/figures must be referenced by the report.

    Eight were not: four cost real compute every run and were never shown, four were
    stale artefacts of an older pipeline still tracked in git (FLAWS-N37).
    """
    root = Path(__file__).resolve().parent
    figures_root = figures_root or (root / "figures")
    tex_path = tex_path or (root / "model_risk_report.tex")
    if not figures_root.exists() or not tex_path.exists():
        return
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    referenced = {
        m.replace("figures/", "").replace("\\", "/")
        for m in re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    }
    orphans = sorted(
        p.relative_to(figures_root).as_posix()
        for p in figures_root.rglob("*.png")
        if p.relative_to(figures_root).as_posix() not in referenced
    )
    _check(
        not orphans,
        f"{len(orphans)} figure(s) exist on disk but are referenced by no \\includegraphics: "
        + ", ".join(orphans)
        + " — either add them to the report or stop generating them",
        failures,
    )


def check_model_claims_vs_code(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """Ban methodological claims the codebase does not implement.

    Each phrase below appeared in a shipped report describing a technique that is nowhere
    in the pipeline (FLAWS-N8, N20, N30).
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    banned = {
        "Vector Autoregressive": "no VAR model exists; the macro link is an OLS regression",
        "Monotonic trends are enforced across bins using isotonic regression":
            "neither binner uses isotonic regression",
        "A logistic regression models the probability of a zero-loss outcome":
            "the cure stage is a GradientBoostingClassifier, not a logistic regression",
    }
    for phrase, why in banned.items():
        _check(
            phrase not in text,
            f"Report claims {phrase!r} but {why}",
            failures,
        )


def check_vintage_drift_claim(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """A "consistently below X" claim must be true of the table beside it.

    The report asserted 2016--2018 PD ratios "consistently below 0.85" directly above a
    table in which not one row was below 0.85 (FLAWS-N7).
    """
    rows = metrics.get("pd_backtest_vintage") or []
    if not rows:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")

    def _ratios_for_era(lo: int | None, hi: int | None) -> list[float]:
        out = []
        for r in rows:
            ratio = r.get("pd_ratio")
            if ratio is None:
                continue
            vintage = str(r.get("vintage", ""))
            if lo is not None and hi is not None:
                if not (vintage[:4].isdigit() and lo <= int(vintage[:4]) <= hi):
                    continue
            out.append(float(ratio))
        return out

    # "...in the 2016--2018 vintages (PD Ratio consistently below 0.85)" — the claim is
    # about a named era, so it must be tested against that era's rows, not the whole table.
    pattern = (
        r"(?:(\d{4})--(\d{4})\s+vintages[^.]{0,120}?)?"
        r"PD Ratio consistently (below|above) ([\d.]+)"
    )
    for match in re.finditer(pattern, text):
        lo_s, hi_s, direction, thresh_s = match.groups()
        threshold = float(thresh_s)
        lo = int(lo_s) if lo_s else None
        hi = int(hi_s) if hi_s else None
        ratios = _ratios_for_era(lo, hi)
        if not ratios:
            continue
        era = f"{lo}-{hi}" if lo else "all"
        if direction == "below":
            ok = all(r < threshold for r in ratios)
            worst = max(ratios)
        else:
            ok = all(r > threshold for r in ratios)
            worst = min(ratios)
        _check(
            ok,
            f"Report claims PD ratios are 'consistently {direction} {threshold}' for the "
            f"{era} vintages, but {sum(1 for r in ratios if (r >= threshold) == (direction == 'below'))} "
            f"of {len(ratios)} rows contradict it (worst {worst:.3f})",
            failures,
        )


def check_challenger_attribution(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """A model must not be credited with another model's score.

    Section 3.6 attributed the weighted ensemble's OOT AUC of 0.7030 to LightGBM, whose
    own figure was 0.7027 (FLAWS-N34).
    """
    rows = metrics.get("ml_benchmark_comparison") or []
    if not rows:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    by_auc: dict[str, str] = {}
    for r in rows:
        auc = r.get("oot_auc")
        if auc is None:
            continue
        by_auc.setdefault(f"{float(auc):.4f}", str(r.get("model", "")))
    for match in re.finditer(r"(LightGBM|XGBoost|Random Forest)[^.]{0,80}?([01]\.\d{4})", text):
        named, quoted = match.group(1), match.group(2)
        owner = by_auc.get(quoted)
        if owner is None:
            continue
        _check(
            named.lower().replace(" ", "") in owner.lower().replace(" ", ""),
            f"Report attributes OOT AUC {quoted} to {named}, but that figure belongs to "
            f"{owner}",
            failures,
        )


def check_phase_failures_surfaced(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """If phases dropped out, the report must say so somewhere (FLAWS-N39)."""
    pf = metrics.get("phase_failures") or []
    if not pf:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    _check(
        "Enhancement phases dropped" in text,
        f"{len(pf)} phase(s) failed but the report contains no disclosure of it",
        failures,
    )


def check_ecl_sensitivity_responds(metrics: dict, failures: list[str]) -> None:
    """The ECL macro-sensitivity grid must actually respond to the macro factor.

    Found by eye, not by any guard: the tornado chart showed an identical ECL at every Z
    from -2.0 to +2.0. The macro overlay had been silently disabled -- the Vasicek shock
    was being anchored on the 12-month cumulative hazard, which is ~0 for every loan in a
    model that places each default in the loan's final month, so the scaling factor
    collapsed to 1. Every existing check passed throughout, because each number was
    internally consistent; they were consistently the baseline.

    Requires a material spread across the grid, and monotonicity in the documented
    direction (Z < 0 is adverse, so ECL must fall as Z rises).
    """
    rows = metrics.get("ecl_sensitivity") or []
    if len(rows) < 3:
        return
    pts = sorted(
        ((float(r["macro_shock"]), float(r["total_ecl"])) for r in rows
         if r.get("macro_shock") is not None and r.get("total_ecl") is not None),
        key=lambda p: p[0],
    )
    if len(pts) < 3:
        return
    ecls = [e for _, e in pts]
    lo, hi = min(ecls), max(ecls)
    _check(
        hi > 0 and (hi - lo) / hi > 0.01,
        f"ECL macro sensitivity is flat across Z: every scenario returns "
        f"{hi:,.0f} (spread {100 * (hi - lo) / max(hi, 1):.4f}%). The macro overlay is "
        "not reaching the term structure",
        failures,
    )
    adverse = [e for z, e in pts if z < 0]
    benign = [e for z, e in pts if z > 0]
    if adverse and benign:
        _check(
            min(adverse) > max(benign),
            "ECL does not fall monotonically as Z rises: adverse scenarios (Z<0) must "
            f"produce higher ECL than benign ones (min adverse {min(adverse):,.0f} vs "
            f"max benign {max(benign):,.0f})",
            failures,
        )


def check_page_count(failures: list[str], pdf_path: Path | None = None,
                     lo: int = 36, hi: int = 41) -> None:
    """Keep the PDF inside its agreed length envelope.

    Raised 40 -> 41 for the July 2026 review round, which added four limitation items
    (out-of-time sample selection, exposure booked on terminated loans, the training-window
    macro fit, and the recalibrator's vintage scope) plus the macro-provenance,
    Hosmer-Lemeshow-subsample and DeLong-covariance sentences. That is one page of
    disclosure the audit specifically asked for; the alternative was deleting it to protect
    a page count, which inverts what this check is for. The band is a bloat guard, not a
    disclosure cap -- if a future round needs another page, widen it deliberately and say
    what bought the page, as here.

    The envelope is 36-40, not the 33-35 originally sketched in the internal review log. That target was
    costed against roughly 0.8 pages of new disclosure; the Round 3 fixes in fact required
    substantially more -- the four-stage selection funnel, the PD-horizon derivation, the
    proportional-hazards shock equation, the EL-to-ECL reconciliation, the CSI table and
    five methodological disclosures. Every mechanical saving was taken (annual vintage
    rows, top-15 IV ranking, two representative points ladders, paired EDA figures,
    withdrawn stage-migration table, merged assumptions list, tighter spacing and
    margins), which brought 42 back to 39. Cutting further would mean deleting the
    disclosures the audit exists to add, so the band was widened instead.
    """
    pdf_path = pdf_path or (Path(__file__).resolve().parent / "model_risk_report.pdf")
    log_path = pdf_path.with_suffix(".log")
    if not log_path.exists():
        return
    log = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"\((\d+) pages", log)
    if not matches:
        return
    pages = int(matches[-1])
    _check(
        lo <= pages <= hi,
        f"report is {pages} pages, outside the agreed {lo}-{hi} page envelope",
        failures,
    )


def check_stage1_ecl_nonzero(metrics: dict, failures: list[str]) -> None:
    """A populated Stage 1 with a zero provision means the 12-month ECL leg is inoperative.

    The hazard panel used to pin every default event to the loan's final month with no
    censoring, so the fitted 12-month hazard was ~0 and Stage 1 ECL came out at exactly
    $0.00 across 483,685 loans and $1.24bn of exposure — printed in the report's own
    reconciliation table, and unguarded.
    """
    recon = metrics.get("ecl_reconciliation") or {}
    ecl_by_stage = recon.get("ecl_by_stage") or {}
    n_by_stage = recon.get("n_by_stage") or {}
    if not ecl_by_stage or not n_by_stage:
        return
    n_s1 = int(n_by_stage.get("s1", 0))
    if n_s1 <= 0:
        return
    _check(
        float(ecl_by_stage.get("s1", 0.0)) > 0.0,
        f"Stage 1 ECL is {float(ecl_by_stage.get('s1', 0.0)):,.2f} across {n_s1:,} Stage 1 "
        "loans — the 12-month ECL leg of the three-stage model is producing no provision "
        "at all (check the hazard model's duration/censoring construction)",
        failures,
    )


def check_hazard_pd12m_nondegenerate(metrics: dict, failures: list[str]) -> None:
    """The hazard model's 12-month PD must be a probability, not a rounding artefact."""
    recon = metrics.get("ecl_reconciliation") or {}
    by_stage = recon.get("mean_pd_12m_by_stage") or {}
    n_by_stage = recon.get("n_by_stage") or {}
    if not by_stage:
        return
    for stage, mean_pd in by_stage.items():
        if int(n_by_stage.get(stage, 0)) <= 0:
            continue
        _check(
            float(mean_pd) > 1e-6,
            f"mean 12-month PD for {stage} is {float(mean_pd):.3g} — degenerate. The "
            "hazard term structure is producing no default probability inside the first "
            "twelve months",
            failures,
        )
    basis = metrics.get("hazard_duration_basis")
    if basis is not None:
        _check(
            basis == "payments_observed_censored",
            f"hazard model duration basis is {basis!r}, not 'payments_observed_censored': "
            "the person-period panel fell back to the uncensored full-term construction",
            failures,
        )


def check_cutoff_el_nonzero(metrics: dict, failures: list[str]) -> None:
    """Every approved book carries credit losses; a zero EL column is a plumbing bug.

    `run_ifrs9_ecl` used to overwrite `pd_12m` on the ECL frame with the hazard model's
    (degenerate) 12-month PD, and the Phase 9 sweep reads that column — so expected_loss
    came out at exactly 0.0 for every cutoff and the published RAROC was
    revenue-minus-costs with no credit loss.
    """
    rows = [
        r for r in (metrics.get("cutoff_strategy_table") or [])
        if float(r.get("approval_rate", 0.0)) > 0.0
    ]
    if not rows:
        return
    zero_rows = [r for r in rows if float(r.get("expected_loss", 0.0)) <= 0.0]
    _check(
        not zero_rows,
        f"{len(zero_rows)} of {len(rows)} non-empty cutoffs charge an expected loss of "
        "zero; the cutoff economics are not netting out credit losses at all "
        f"(first offender: cutoff {zero_rows[0].get('cutoff') if zero_rows else 'n/a'})",
        failures,
    )

    # A zero test alone is too weak: a PD of 1e-20 against a $1bn book still yields a
    # strictly positive EL. Anchor it instead. At full approval the sweep's EL is
    # PD_12m x LGD x EAD over (essentially) the same population as the Phase 6 portfolio
    # Expected Loss, so the two must land in the same ballpark. On the degenerate run they
    # were $0.00 and $279.7m.
    total_el = float(metrics.get("total_el", 0.0))
    full_rows = [r for r in rows if float(r.get("approval_rate", 0.0)) >= 0.999]
    if not full_rows or total_el <= 0.0:
        return
    sweep_el = float(max(full_rows, key=lambda r: float(r["approval_rate"]))["expected_loss"])
    ratio = sweep_el / total_el
    _check(
        0.5 <= ratio <= 2.0,
        f"cutoff sweep charges {sweep_el:,.0f} of expected loss at full approval against a "
        f"portfolio Expected Loss of {total_el:,.0f} (ratio {ratio:.3g}) — the two are the "
        "same quantity on the same book and cannot disagree by this much; the sweep is "
        "reading a different PD column",
        failures,
    )


def check_cutoff_raroc_prose(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """Prose about the shape of the RAROC grid must match the grid.

    `check_cutoff_corner_direction` bans "full approval" wording when the argmax is an
    exclusive corner, but has no symmetric guard — so the opposite error shipped: the
    report asserted "every cutoff on the 400--800 grid returns a negative RAROC" on a run
    where every grid RAROC was positive, and claimed the profit and RAROC argmaxes
    "coincide" when they sat at 400 and 610.
    """
    rows = [
        r for r in (metrics.get("cutoff_strategy_table") or [])
        if float(r.get("approval_rate", 0.0)) > 0.0
    ]
    if not rows:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")

    n_negative = sum(1 for r in rows if float(r.get("raroc", 0.0)) < 0.0)
    if n_negative < len(rows):
        _check(
            "returns a negative RAROC" not in text,
            f"Report states every cutoff on the grid returns a negative RAROC, but only "
            f"{n_negative} of {len(rows)} non-empty cutoffs do",
            failures,
        )

    profit_argmax = metrics.get("cutoff_profit_argmax") or {}
    raroc_max = metrics.get("cutoff_raroc_max") or {}
    if profit_argmax and raroc_max:
        coincide = int(profit_argmax.get("cutoff", -1)) == int(raroc_max.get("cutoff", -2))
        if not coincide:
            _check(
                "RAROC-maximising cutoff coincide" not in text,
                "Report says the profit-maximising and RAROC-maximising cutoffs coincide, "
                f"but they sit at {profit_argmax.get('cutoff')} and "
                f"{raroc_max.get('cutoff')}",
                failures,
            )


def check_no_hardcoded_metric_defaults(failures: list[str], renderer_path: Path | None = None) -> None:
    """The renderer must not carry hand-typed fallbacks for numbers it prints.

    There were 66 ``metrics.get("<key>", <constant>)`` lookups in render_latex.py, 34 of
    them holding a non-zero figure from a superseded run and nine keys holding *different*
    constants at different call sites. A single absent key could therefore print two
    different values for the same quantity in two sections, silently, and every one of
    those constants would read as a real result. Headline numbers now go through
    ``_num()``, which raises for required keys and yields a visible "n/a" otherwise.
    """
    path = renderer_path or (Path(__file__).resolve().parent / "render_latex.py")
    if not path.exists():
        return
    src = path.read_text(encoding="utf-8")
    offenders = re.findall(
        r'metrics\.get\(\s*"([^"]+)"\s*,\s*[0-9][0-9_.eE+-]*\s*\)', src
    )
    _check(
        not offenders,
        "render_latex.py still falls back to hand-typed constants for metrics "
        f"{sorted(set(offenders))} — a stale run's numbers can reach the PDF unnoticed. "
        "Use _num(metrics, key).",
        failures,
    )


def check_no_unpublished_doc_references(failures: list[str], root: Path | None = None) -> None:
    """Source comments must not cite files that are not in the repository.

    163 references across 30 files pointed at the two internal review logs, both
    gitignored. Every one of them was a dead end for anyone reading the
    published repository. Citations now use stable IDs resolved by ``docs/FINDINGS.md``.
    """
    base = root or Path(__file__).resolve().parent.parent
    # Assembled at runtime so this checker does not match itself.
    pattern = re.compile("|".join([
        "docs/" + "AUDIT" + r"\.md",
        "Flaws" + r"\.md",
    ]))
    offenders: list[str] = []
    for sub in ("src", "reports", "tests", "config", "scripts"):
        d = base / sub
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if pattern.search(text):
                offenders.append(str(path.relative_to(base)))
    _check(
        not offenders,
        f"source files cite unpublished internal notes: {sorted(offenders)[:5]} "
        "(use a stable ID from docs/FINDINGS.md instead)",
        failures,
    )


def check_required_metrics_present(metrics: dict, failures: list[str]) -> None:
    """Every headline key the report prints must exist in metrics.json."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from render_latex import _REQUIRED_METRICS  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - renderer not importable in this context
        return
    missing = sorted(k for k in _REQUIRED_METRICS if metrics.get(k) is None)
    _check(
        not missing,
        f"metrics.json is missing headline key(s) {missing}; the report would have "
        "printed a fallback constant for them before this guard existed",
        failures,
    )


def run_metric_checks(metrics: dict) -> None:
    """Run all metrics.json identity checks; raise QAError listing every failure."""
    failures: list[str] = []
    check_rwa_density(metrics, failures)
    check_ecl_coverage(metrics, failures)
    check_capital_identity(metrics, failures)
    check_cutoff_optimum(metrics, failures)
    check_scenario_dr(metrics, failures)
    check_scenario_ordering(metrics, failures)
    check_irb_sa_direction(metrics, failures)
    check_vintage_pd_ratio(metrics, failures)
    check_es_ge_var(metrics, failures)
    check_benchmarks_sourced(failures)
    check_challenger_feature_parity(metrics, failures)
    check_scenario_axes_separate(metrics, failures)
    # Round 3 (the internal review log)
    check_binner_is_optbinning(metrics, failures)
    check_no_phase_failures(metrics, failures)
    check_recalibration_gate_chronology(metrics, failures)
    check_calibration_table_direction(metrics, failures)
    check_pd_horizon_consistency(metrics, failures)
    check_stress_direction(metrics, failures)
    check_vintage_calibration_band(metrics, failures)
    check_feature_selection_stages(metrics, failures)
    check_ecl_sensitivity_responds(metrics, failures)
    # Round 4: the degenerate 12-month PD chain
    check_stage1_ecl_nonzero(metrics, failures)
    check_hazard_pd12m_nondegenerate(metrics, failures)
    check_cutoff_el_nonzero(metrics, failures)
    # AUDIT_FINDINGS round: no rendered number may originate from a hand-typed default
    check_no_hardcoded_metric_defaults(failures)
    check_required_metrics_present(metrics, failures)
    check_no_unpublished_doc_references(failures)
    if failures:
        raise QAError(
            "Report QA failed (%d issue(s)):\n  - %s"
            % (len(failures), "\n  - ".join(failures))
        )
    print(f"QA metric checks passed ({len(metrics)} metric keys audited).")


def run_tex_checks(tex_path: str | Path, metrics: dict | None = None) -> None:
    """Run generated-.tex guards; raise QAError on any unreplaced placeholder,
    dangling citation, or text/metric inconsistency."""
    failures: list[str] = []
    path = Path(tex_path)
    check_no_unreplaced_vars(path, failures)
    check_no_unescaped_percent(path, failures)
    check_no_fabricated_benchmark(path, failures)
    check_citations_resolve(path, failures)
    if metrics is not None:
        check_lgd_r2_consistency(path, metrics, failures)
        check_recalibration_claim(path, metrics, failures)
        check_recalibration_applied(path, metrics, failures)
        check_cutoff_corner_direction(path, metrics, failures)
        check_capital_charge_rate(path, metrics, failures)
        check_population_counts(path, metrics, failures)
        # Round 3 (the internal review log)
        check_model_claims_vs_code(path, metrics, failures)
        check_vintage_drift_claim(path, metrics, failures)
        check_challenger_attribution(path, metrics, failures)
        check_phase_failures_surfaced(path, metrics, failures)
        # Round 4
        check_cutoff_raroc_prose(path, metrics, failures)
    check_no_orphan_figures(failures, tex_path=path)
    check_page_count(failures)
    if failures:
        raise QAError(
            "Report QA failed (%d issue(s)):\n  - %s"
            % (len(failures), "\n  - ".join(failures))
        )
    print(f"QA tex checks passed for {tex_path}.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    with open(root / "outputs" / "metrics.json", encoding="utf-8") as f:
        _metrics = json.load(f)
    run_metric_checks(_metrics)
    tex = root / "reports" / "model_risk_report.tex"
    if tex.exists():
        run_tex_checks(tex, _metrics)
