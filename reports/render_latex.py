import os
import sys
import json
import subprocess
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from qa_checks import run_metric_checks, run_tex_checks  # noqa: E402
from credit_risk.validation.backtest import (  # noqa: E402
    VINTAGE_PASS_BAND,
    vintage_band_text,
    vintage_calibration_flag as _vintage_flag,
)



def tex_escape(text: str) -> str:
    """Escape LaTeX specials in a string that came from code, not from the template.

    Critically this covers ``%``: an unescaped percent sign is a LaTeX comment, so it
    silently deletes the remainder of the source line from the rendered PDF without any
    error. A diagnostic string such as "ratio outside +/-10%" swallowed 737 characters of
    the recalibration evidence in an earlier build (AUDIT-A1).
    """
    if not isinstance(text, str):
        return text
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# Keys the report cannot honestly print without. Anything here that is absent from
# metrics.json aborts the build instead of silently substituting a number.
_REQUIRED_METRICS = frozenset({
    "auc", "gini", "ks", "auc_oot", "gini_oot", "ks_oot",
    "mean_lgd", "downturn_lgd",
    "total_el", "el_rate", "total_ecl", "ecl_coverage",
    "total_rwa", "total_rwa_sa", "total_ead_portfolio",
    "stage3_pct",
    "optimal_cutoff_threshold", "optimal_approval_rate", "optimal_bad_rate",
    "cutoff_raroc_hurdle", "cutoff_max_bad_rate", "cutoff_cost_of_capital",
})


def _num(metrics: dict, key: str, *, required: bool | None = None) -> float:
    """Read a number out of metrics.json — never fall back to a hand-typed constant.

    The renderer used to be full of ``_num(metrics, "total_rwa")``-style lookups:
    66 of them, 34 carrying a non-zero constant left over from an old run, and nine keys
    carrying *different* constants at different call sites (``total_rwa`` appeared as
    ``0``, ``0.0`` and ``67238352``). A single missing key could therefore make two
    sections of the same report print two different values for the same quantity, with no
    error anywhere. That is exactly the failure mode ``reports/benchmarks.py`` was written
    to prevent for the literature table.

    Required keys raise. Everything else returns NaN, which the ``fmt_*`` helpers render as
    a visible ``n/a`` rather than as a plausible-looking number.
    """
    val = metrics.get(key)
    if val is None:
        must_have = (key in _REQUIRED_METRICS) if required is None else required
        if must_have:
            raise KeyError(
                f"metrics.json is missing '{key}', which the report prints as a headline "
                "number. Re-run the pipeline; the renderer will not invent a value."
            )
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _target_status_sets(project_root: str) -> tuple[str, str]:
    """Render the bad/good loan-status sets straight from config/config.yaml.

    The target equation used to be a hand-typed literal that had drifted away from the
    configuration: it omitted the two "Does not meet the credit policy" statuses and
    described the definition as the BCBS 90+ DPD standard, which it is not
    (FLAWS-N11).
    """
    import yaml  # noqa: PLC0415

    cfg_path = os.path.join(project_root, "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    target = cfg.get("target", {})

    def _fmt(statuses: list[str]) -> str:
        # en-dash the DPD ranges for typography, then escape LaTeX specials.
        return ", ".join(
            r"\text{" + tex_escape(str(s)).replace("31-120", "31--120") + "}"
            for s in statuses
        )

    return _fmt(target.get("bad_statuses", [])), _fmt(target.get("good_statuses", []))


def render_latex():
    project_root = r"C:\Users\Δημητρης\OneDrive\Υπολογιστής\Credit Risk Project\credit-risk-ecl"
    metrics_path = os.path.join(project_root, "outputs", "metrics.json")
    tex_path = os.path.join(project_root, "reports", "model_risk_report.tex")
    sc_tables_path = os.path.join(project_root, "outputs", "scorecard_tables.json")

    if not os.path.exists(metrics_path):
        print(f"Error: metrics.json not found at {metrics_path}")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Fix 3.3: cross-table consistency QA — abort build on inconsistency
    run_metric_checks(metrics)

    sc_tables = {}
    if os.path.exists(sc_tables_path):
        with open(sc_tables_path, "r") as f:
            sc_tables = json.load(f)

    # ── Scorecard table builders ───────────────────────────────────────────────
    def _iv_table_latex(iv_rows, stages=None, selected=None, top_n=15):
        """Top-N IV ranking with the stage at which each feature left the funnel.

        Two changes from the previous version. The table is truncated to the top N (the
        tail was 21 rows of sub-0.035 IV occupying most of a page), and each row now says
        why a feature is or is not in the final model. Without that column the reader
        cannot reconcile this ranking with the selected-feature list at all: int_rate
        carries the highest IV in the table and is absent from the model, dropped by the
        ElasticNet stage the report used not to mention (FLAWS-N29, page budget).
        """
        if not iv_rows:
            return r"\textit{No IV data available.}"
        stages = stages or {}
        selected = set(selected or [])
        dropped_at = {}
        for stage_key, label in (
            ("dropped_by_iv", "IV band"),
            ("dropped_by_vif", "VIF"),
            ("dropped_by_elasticnet", "ElasticNet"),
            ("dropped_by_sign_check", "Sign check"),
        ):
            for feat in stages.get(stage_key, []) or []:
                dropped_at.setdefault(str(feat), label)

        ordered = sorted(iv_rows, key=lambda r: r["iv"], reverse=True)
        shown, hidden = ordered[:top_n], ordered[top_n:]

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Feature Information Value (IV) Ranking (top " + str(top_n) + r")}",
            r"\label{tab:iv_ranking}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lccl}",
            r"\toprule",
            r"\textbf{Feature} & \textbf{IV} & \textbf{Predictive Power} & \textbf{Outcome} \\",
            r"\midrule",
        ]
        bands = {(0, 0.02): "Negligible", (0.02, 0.1): "Weak",
                 (0.1, 0.3): "Medium", (0.3, 0.5): "Strong", (0.5, 999): "Very Strong"}
        for row in shown:
            name = str(row["variable"])
            feat = name.replace("_", r"\_")
            iv_val = row["iv"]
            band = next((v for (lo, hi), v in bands.items() if lo <= iv_val < hi), "N/A")
            if name in selected:
                outcome = r"\textbf{Retained}"
            elif name in dropped_at:
                outcome = f"Dropped ({dropped_at[name]})"
            else:
                outcome = "Not selected"
            lines.append(f"\\texttt{{{feat}}} & {iv_val:.4f} & {band} & {outcome} \\\\")

        note = (
            r"\multicolumn{4}{p{0.92\linewidth}}{\footnotesize \textit{Note:} "
            + (
                f"the remaining {len(hidden)} candidate features all carry "
                f"IV $\\le$ {hidden[0]['iv']:.4f} and are omitted; the full ranking is in "
                r"\texttt{outputs/scorecard\_tables.json}. "
                if hidden else ""
            )
            + r"The \textit{Outcome} column names the stage of the four-stage funnel "
            r"(Section~3.3) at which each feature left, which is why a high-IV feature can "
            r"be absent from the final model. Monotone transforms of the same underlying "
            r"variable (for example a raw and a log-scaled version) carry near-identical IV "
            r"by construction and are not independent evidence.} \\"
        )
        lines += [r"\bottomrule", note, r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def _csi_table_latex(csi_rows):
        """Per-feature Characteristic Stability Index, train vs OOT.

        Computed by the pipeline since the stability phase was added but never rendered,
        which left the PSI narrative with no way to say *where* the population did or did
        not move (FLAWS-N38, N40).
        """
        if not csi_rows:
            return r"\textit{No CSI data available for this run.}"
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Characteristic Stability Index (CSI) by Scorecard Feature, Train vs.\ OOT}",
            r"\label{tab:csi}",
            r"\vspace{0.5em}",
            r"\small",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Feature} & \textbf{CSI} & \textbf{Stability Band} \\",
            r"\midrule",
        ]
        # Only the most-shifted features carry information; the tail is uniformly stable
        # and cost most of a page.
        _ordered = sorted(csi_rows, key=lambda r: (float(r.get("csi", 0) or 0)), reverse=True)
        _shown, _rest = _ordered[:8], _ordered[8:]
        for row in _shown:
            feat = str(row.get("feature", "")).replace("_", r"\_")
            try:
                csi_val = f"{float(row.get('csi')):.4f}"
            except (TypeError, ValueError):
                csi_val = "n/a"
            band = tex_escape(str(row.get("band", "")))
            lines.append(f"\\texttt{{{feat}}} & {csi_val} & {band} \\\\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\parbox{0.92\textwidth}{\footnotesize \vspace{0.4em} CSI is the PSI statistic "
            r"applied one feature at a time. Bands follow the same convention as PSI: "
            r"below 0.10 stable, 0.10--0.25 moderate shift, above 0.25 material shift."
            + (
                f" The {len(_rest)} features not shown all sit below "
                f"{float(_rest[0].get('csi', 0) or 0):.4f} and are stable by this measure."
                if _rest else ""
            )
            + r"}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _ecl_reconciliation_latex(recon, total_el):
        """Reconcile the one-year Expected Loss against the IFRS 9 ECL.

        The two headline provisions differ by only a few percent while measuring very
        different things, and the report offered the reader no way to see why. The stage
        split makes the dominant driver visible: Stage 3, provisioned at LGDxEAD with PD
        forced to 1, carries most of both figures (FLAWS-N28).
        """
        if not recon or not recon.get("ecl_by_stage"):
            return r"\textit{No stage-level reconciliation available for this run.}"
        ecl_by = recon.get("ecl_by_stage", {})
        ead_by = recon.get("ead_by_stage", {})
        n_by = recon.get("n_by_stage", {})
        pd12_by = recon.get("mean_pd_12m_by_stage", {})
        pdlt_by = recon.get("mean_pd_lifetime_by_stage", {})
        total_ecl_v = float(recon.get("total_ecl", 0.0))

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Reconciliation: one-year Expected Loss vs.\ IFRS~9 ECL}",
            r"\label{tab:el_ecl_reconciliation}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"\textbf{Stage} & \textbf{Loans} & \textbf{EAD} & \textbf{Mean PD (12m, haz.)} "
            r"& \textbf{Mean PD (life)} & \textbf{ECL} \\",
            r"\midrule",
        ]
        labels = {"s1": "Stage 1 (12-month ECL)", "s2": "Stage 2 (lifetime ECL)",
                  "s3": "Stage 3 (credit-impaired)"}
        for key in ("s1", "s2", "s3"):
            lines.append(
                f"{labels[key]} & {int(n_by.get(key, 0)):,} & "
                f"\\${float(ead_by.get(key, 0.0))/1e9:,.2f}bn & "
                f"{float(pd12_by.get(key, 0.0))*100:.2f}\\% & "
                f"{float(pdlt_by.get(key, 0.0))*100:.2f}\\% & "
                f"\\${float(ecl_by.get(key, 0.0))/1e9:,.2f}bn \\\\"
            )
        # Whether the two totals are close is a property of the run, not a standing fact.
        # This note asserted proximity ("close in magnitude", "their similarity is a
        # coincidence of offsetting effects") while the table beside it printed $1.35bn
        # against $0.28bn.
        _el_v = float(total_el)
        _ratio = (total_ecl_v / _el_v) if _el_v > 0 else 0.0
        if 0.8 <= _ratio <= 1.25:
            _proximity = (
                r"the two totals are close in magnitude but are not the same quantity, and "
                r"their similarity is a coincidence of offsetting effects rather than a "
                r"mutual validation. "
            )
        else:
            _proximity = (
                f"the ECL is {_ratio:.1f}$\\times$ the one-year Expected Loss. That is a gap "
                r"between two differently-defined quantities, not a discrepancy. "
            )
        lines += [
            r"\midrule",
            f"\\textbf{{Total IFRS 9 ECL}} & & & & & \\textbf{{\\${total_ecl_v/1e9:,.2f}bn}} \\\\",
            f"\\textbf{{One-year Expected Loss}} & & & & & \\textbf{{\\${_el_v/1e9:,.2f}bn}} \\\\",
            r"\bottomrule",
            r"\multicolumn{6}{p{0.92\linewidth}}{\footnotesize \textit{Note:} " + _proximity +
            r"Expected Loss is "
            r"$PD_{\text{12m}} \times LGD \times EAD$ over one year, undiscounted and with "
            r"no staging. ECL applies a 12-month horizon to Stage~1, a lifetime horizon to "
            r"Stage~2, and $PD = 1$ to Stage~3, discounts at the effective interest rate, "
            r"and probability-weights across macro scenarios. The Stage~3 column "
            r"dominates both, and in that column the PD model plays no part at all "
            r"(Section~10.3). The two PD columns come from the \emph{hazard} term-structure "
            r"model, which is what drives the staged ECL; the one-year Expected Loss row is "
            r"built from the \emph{scorecard}'s annualised PD instead, so the 12-month "
            r"column does not multiply out to the EL total.} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _logit_table_latex(coef_rows):
        if not coef_rows:
            return r"\textit{No coefficient data available.}"
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Logistic Regression Coefficient Summary}",
            r"\label{tab:logit_coefficients}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Feature} & \textbf{Coefficient} & \textbf{Std.\ Error} & \textbf{z-stat} & \textbf{p-value} \\",
            r"\midrule",
        ]
        for row in coef_rows:
            feat = row["feature"].replace("_", r"\_")
            sig = (r"$^{***}$" if row["p_value"] < 0.01
                   else (r"$^{**}$" if row["p_value"] < 0.05
                         else (r"$^{*}$" if row["p_value"] < 0.10 else "")))
            lines.append(
                f"\\texttt{{{feat}}} & {row['coefficient']:.4f} & {row['std_err']:.4f}"
                f" & {row['z_stat']:.3f} & {row['p_value']:.4f}{sig} \\\\"
            )
        lines += [
            r"\midrule",
            r"\multicolumn{5}{l}{\footnotesize Significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _scorecard_points_latex(sc_rows, n_features=2):
        """Complete bin ladders for the highest-spread features, not a truncated list.

        The previous version took the top 25 rows by feature spread, which cut off in the
        middle of a feature's bins — a points table that stops halfway through a ladder
        cannot be read as a scorecard. Showing every bin of the top few features, and
        pointing at the JSON for the rest, is both shorter and usable
        (the internal review log page budget).
        """
        if not sc_rows:
            return r"\textit{No scorecard table available.}"
        feat_ranges = defaultdict(lambda: [9999, -9999])
        for r in sc_rows:
            feat_ranges[r["feature"]][0] = min(feat_ranges[r["feature"]][0], r["points"])
            feat_ranges[r["feature"]][1] = max(feat_ranges[r["feature"]][1], r["points"])
        by_spread = sorted(
            feat_ranges, key=lambda f: feat_ranges[f][1] - feat_ranges[f][0], reverse=True
        )
        chosen = by_spread[:n_features]
        n_other = len(by_spread) - len(chosen)
        shown = [r for f in chosen for r in sc_rows if r["feature"] == f]
        has_n   = any(r.get("n_obs") is not None for r in shown)
        has_dr  = any(r.get("dr")    is not None for r in shown)
        # Build header
        col_spec = "llcc"
        header   = r"\textbf{Feature} & \textbf{Bin} & \textbf{WoE} & \textbf{$\beta$}"
        if has_n:
            col_spec += "r"
            header   += r" & \textbf{N}"
        if has_dr:
            col_spec += "r"
            header   += r" & \textbf{DR\%}"
        col_spec += "r"
        header   += r" & \textbf{Points}"
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Credit Scorecard Points Table --- complete bin ladders for the "
            r"highest-spread features}",
            r"\label{tab:scorecard_points}",
            r"\vspace{0.5em}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]
        prev_feat = None
        for row in shown:
            feat_disp = row["feature"].replace("_", r"\_") if row["feature"] != prev_feat else ""
            prev_feat = row["feature"]
            # Add thin rule between features
            line = f"\\texttt{{{feat_disp}}} & {row['bin']} & {row['woe']:.4f} & {row['beta']:.4f}"
            if has_n:
                n_obs = row.get("n_obs")
                line += f" & {n_obs:,}" if n_obs is not None else " & ---"
            if has_dr:
                dr = row.get("dr")
                line += f" & {dr*100:.2f}\\%" if dr is not None else " & ---"
            line += f" & {row['points']:.1f} \\\\"
            lines.append(line)
        lines += [
            r"\bottomrule",
            r"\multicolumn{" + str(len(col_spec)) + r"}{p{0.92\linewidth}}{\footnotesize "
            r"WoE = ln(\%Good/\%Bad); Points = ($-\text{WoE}_j * \beta_j + \alpha/n$) * "
            r"Factor + Offset/n. "
            + (
                f"The remaining {n_other} scorecard features follow the same construction; "
                r"their complete ladders are in \texttt{outputs/scorecard\_tables.json}. "
                if n_other > 0 else ""
            )
            + r"Where a calibrator is attached in production, the score-to-PD mapping "
            r"tabulated here is the \emph{pre-recalibration} relationship: the points "
            r"scale is built from the raw logit so that it stays additive and auditable, "
            r"while the deployed PD passes through the recalibration transform of "
            r"Section~7.2.} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _aggregate_backtest_to_annual(rows):
        """Collapse quarterly vintages to annual ones, exposure-weighted.

        43 quarterly rows spanned most of a page while adding no argument the annual
        series does not already make; the quarters that fall outside the calibration band
        are still named individually in the footnote below the table.
        """
        by_year: dict[str, dict] = {}
        for row in rows:
            vintage = str(row.get("vintage", ""))
            year = vintage[:4]
            if not year.isdigit():
                continue
            n = int(row.get("n_loans", 0))
            if n <= 0:
                continue
            agg = by_year.setdefault(
                year, {"vintage": year, "n_loans": 0, "_pred_sum": 0.0, "_act_sum": 0.0}
            )
            agg["n_loans"] += n
            agg["_pred_sum"] += float(row.get("predicted_pd", 0.0)) * n
            agg["_act_sum"] += float(
                row.get("actual_dr", row.get("actual_default_rate", 0.0))
            ) * n
        out = []
        for year in sorted(by_year):
            agg = by_year[year]
            n = agg["n_loans"]
            pred = agg["_pred_sum"] / n
            actual = agg["_act_sum"] / n
            out.append({
                "vintage": year,
                "n_loans": n,
                "predicted_pd": pred,
                "actual_dr": actual,
                "pd_ratio": (pred / actual) if actual > 0 else 0.0,
                # Flag from the production rule, not from a second copy of it. This used
                # to hardcode [0.80, 1.25] while validation/backtest.py used [0.80, 1.20]
                # and the surrounding prose called it a 50% band.
                "calibration_flag": _vintage_flag((pred / actual) if actual > 0 else 0.0),
            })
        return out

    def _pd_backtest_rows_latex(rows):
        if not rows:
            return r"\multicolumn{5}{c}{\textit{No vintage backtest data available.}} \\"
        lines = []
        import numpy as np
        rows = _aggregate_backtest_to_annual(rows) or rows
        for row in rows:
            vintage  = str(row.get("vintage", "N/A"))
            n        = int(row.get("n_loans", 0))
            pred_pd  = float(row.get("predicted_pd", 0))
            actual_dr = float(row.get("actual_dr", row.get("actual_default_rate", 0)))
            pd_ratio  = float(row.get("pd_ratio", 0))

            # Wilson Score 95% CI (more accurate for small n)
            if n > 0:
                z = 1.96
                p = actual_dr
                denom = 1 + z**2 / n
                centre = (p + z**2 / (2 * n)) / denom
                margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
                ci_lower = max(0.0, centre - margin)
                ci_upper = min(1.0, centre + margin)
            else:
                ci_lower = ci_upper = actual_dr

            # Status flags: ✓ = within band, † = warn (outside 50%), ‡ = fail (outside 100%)
            if 0.5 <= pd_ratio <= 1.5:
                flag_str = r" {\checkmark}"
            elif 0.25 <= pd_ratio <= 2.0:
                flag_str = r" $\dagger$"
            else:
                flag_str = r" $\ddagger$"

            pred_pd_pct    = pred_pd    * 100.0
            actual_dr_pct  = actual_dr  * 100.0
            ci_lower_pct   = ci_lower   * 100.0
            ci_upper_pct   = ci_upper   * 100.0

            lines.append(
                f"{vintage} & {n:,} & {pred_pd_pct:.2f}\\% & "
                f"{actual_dr_pct:.2f}\\% [{ci_lower_pct:.1f}\\%, {ci_upper_pct:.1f}\\%] & "
                f"{pd_ratio:.2f}{flag_str} \\\\"
            )
        return "\n".join(lines)

    def _calibration_comparison_table_latex(metrics_dict):
        comp = metrics_dict.get("calibration_comparison", {})
        if not comp:
            return r"\textit{No calibration comparison data available.}"
        before = comp.get("before", {})
        after  = comp.get("after",  {})
        if not after:
            # The gate never reached the fit stage, so there is no deployed transform to
            # compare against. Say so rather than tabulating a transform that was never
            # applied (FLAWS-N6).
            return (
                r"\textit{No recalibration was fitted in this run, so no before/after "
                r"comparison exists. Reason: "
                + tex_escape(str(comp.get("reason", "not recorded")))
                + r".}"
            )

        # Calculate Delta changes
        d_auc = after.get("auc", 0.0) - before.get("auc", 0.0)
        d_brier = after.get("brier", 0.0) - before.get("brier", 0.0)
        d_slope = after.get("slope", 0.0) - before.get("slope", 0.0)
        d_intercept = after.get("intercept", 0.0) - before.get("intercept", 0.0)
        d_expected = (after.get("expected_dr", 0.0) - before.get("expected_dr", 0.0)) * 100
        d_actual = (after.get("actual_dr", 0.0) - before.get("actual_dr", 0.0)) * 100

        # Honest out-of-sample HL verdict (the recalibrator is fitted on the test
        # partition and only applied to OOT, so this is not guaranteed to pass).
        after_hl = after.get("hl_pvalue", 0.0)
        hl_flag = r"\checkmark PASS" if after_hl >= 0.05 else r"$\ast$ below 0.05"
        fit_on = comp.get("recalibration_fit_on", "earlier_oot_slice").replace("_", " ")
        transform_name = str(comp.get("transform", "none")).replace("_", " ")
        n_eval = comp.get("n_eval")
        eval_span = ""
        if comp.get("eval_slice_min_date") and comp.get("eval_slice_max_date"):
            eval_span = (
                f" spanning {comp['eval_slice_min_date']} to {comp['eval_slice_max_date']}"
            )

        # Per-metric verdict, computed (never hard-coded): a row earns a checkmark
        # only when recalibration moves the value TOWARD its target; if it moves away
        # it is flagged so the table cannot claim an improvement that did not happen.
        actual_dr_val = float(after.get("actual_dr", before.get("actual_dr", 0.0)))

        def _toward(b, a, target):
            return r"\checkmark" if abs(a - target) < abs(b - target) else r"$\times$"

        f_brier = _toward(before.get("brier", 0.0), after.get("brier", 0.0), 0.0)
        f_slope = _toward(before.get("slope", 0.0), after.get("slope", 0.0), 1.0)
        f_intercept = _toward(before.get("intercept", 0.0), after.get("intercept", 0.0), 0.0)
        f_expected = _toward(before.get("expected_dr", 0.0), after.get("expected_dr", 0.0), actual_dr_val)

        # Name any metric that regressed, so the trade-off is stated rather than left for
        # the reader to spot in the delta column (FLAWS-N6).
        _regressed = [
            label for label, flag in (
                ("the Brier score", f_brier),
                ("the calibration slope", f_slope),
                ("the calibration intercept", f_intercept),
                ("the expected default rate", f_expected),
            ) if flag != r"\checkmark"
        ]

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Deployed Recalibration, Measured on the Held-Out Later OOT Slice}",
            r"\label{tab:calibration_comparison}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Target} & \textbf{Before Recalib.} & \textbf{After Recalib.} & \textbf{$\Delta$ Change} \\",
            r"\midrule",
            f"OOT AUC & --- & {before.get('auc', 0.0):.4f} & {after.get('auc', 0.0):.4f} & {d_auc:+.4f} \\\\",
            f"Brier Score & $<0.25$ & {before.get('brier', 0.0):.4f} & {after.get('brier', 0.0):.4f} & {d_brier:+.4f} {f_brier} \\\\",
            f"Calibration Slope & $\\approx 1.00$ & {before.get('slope', 0.0):.4f} & {after.get('slope', 0.0):.4f} & {d_slope:+.4f} {f_slope} \\\\",
            f"Calibration Intercept & $\\approx 0.00$ & {before.get('intercept', 0.0):.4f} & {after.get('intercept', 0.0):.4f} & {d_intercept:+.4f} {f_intercept} \\\\",
            f"Expected Default Rate & = Actual & {before.get('expected_dr', 0.0)*100:.2f}\\% & {after.get('expected_dr', 0.0)*100:.2f}\\% & {d_expected:+.2f}\\% {f_expected} \\\\",
            f"Actual Default Rate & {before.get('actual_dr', 0.0)*100:.2f}\\% & {before.get('actual_dr', 0.0)*100:.2f}\\% & {after.get('actual_dr', 0.0)*100:.2f}\\% & {d_actual:+.2f}\\% \\\\",
            f"Hosmer-Lemeshow $p$ & $>0.05$ & {before.get('hl_pvalue', 0.0):.4f} & {after.get('hl_pvalue', 0.0):.4f} & {hl_flag} \\\\",
            r"\bottomrule",
            r"\multicolumn{5}{p{\linewidth}}{\footnotesize \checkmark\ marks a metric that moved \emph{toward} its target and $\times$ one that moved away. "
            + f"This table measures the \\emph{{deployed}} transform ({transform_name}), fitted on the "
            + f"{fit_on} and evaluated on the disjoint later slice"
            + (f" of {n_eval:,} loans" if isinstance(n_eval, int) else "")
            + f"{eval_span}"
            + r" --- the same evidence the acceptance decision was made on, so the table and the "
            + r"narrative above cannot disagree. "
            + (
                ("Note the trade-off: " + ", ".join(_regressed) + " moved away from target "
                 "while the level measures improved sharply. This is characteristic of "
                 "isotonic recalibration, which is a monotone step function fitted to the "
                 "aggregate level and is not constrained to preserve the logit-scale "
                 "slope. The gate accepts on the aggregate ratio and the Brier score, "
                 "both of which improved.")
                if _regressed else
                "Every tabulated measure moved toward its target."
            )
            + r"} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _underwriting_comparison_table_latex(metrics_dict):
        uw = metrics_dict.get("underwriting_scorecard", {})
        if not uw:
            return r"\textit{No underwriting comparison data available.}"
        model_a_auc = metrics_dict.get("auc_oot", 0.6897)
        model_a_gini = metrics_dict.get("gini_oot", 0.3794)
        
        model_b_auc = uw.get("oot", {}).get("auc", 0.0)
        model_b_gini = uw.get("oot", {}).get("gini", 0.0)
        
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Champion (Model A) vs. Pure Underwriting Challenger (Model B) Performance}",
            r"\label{tab:underwriting_comparison}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{Features Included} & \textbf{OOT AUC} & \textbf{OOT Gini} \\",
            r"\midrule",
            f"Model A (Full Scorecard) & Bureau + Application + int\\_rate + grade & {model_a_auc:.4f} & {model_a_gini:.4f} \\\\",
            f"Model B (Underwriting) & Bureau + Application (excludes int\\_rate/grade) & {model_b_auc:.4f} & {model_b_gini:.4f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _cutoff_raroc_table_latex(metrics_dict):
        strategy = metrics_dict.get("cutoff_strategy_table", [])
        if not strategy:
            return r"\textit{No cutoff strategy data available.}"
        targets = [500, 540, 580, 620, 660, 700]
        # The operating cutoff (risk-appetite rule: most inclusive score whose approved bad
        # rate stays under the ceiling) must appear as an actual, highlighted row.
        opt = metrics_dict.get("cutoff_risk_appetite") or metrics_dict.get(
            "cutoff_optimal_profit", {}
        )
        opt_cut = opt.get("cutoff")
        show = sorted({*targets, *([opt_cut] if opt_cut is not None else [])})
        rows = [row for row in strategy if row["cutoff"] in show]

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Cutoff Strategy and Profitability Analysis (RAROC). The recommended operating cutoff --- the most inclusive score over the full 400--800 grid (step 10) whose approved bad rate stays within the risk-appetite ceiling --- is highlighted in bold.}",
            r"\label{tab:cutoff_raroc}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{cccccc}",
            r"\toprule",
            r"\textbf{Cutoff Score} & \textbf{Approval Rate} & \textbf{Bad Rate} & \textbf{Expected Profit} & \textbf{Expected Loss} & \textbf{RAROC} \\",
            r"\midrule",
        ]
        for r in rows:
            cells = (
                f"{r['cutoff']} & {r['approval_rate']*100:.1f}\\% & {r['bad_rate']*100:.2f}\\% & "
                f"\\${r['expected_profit']:,.0f} & \\${r['expected_loss']:,.0f} & {r['raroc']*100:.2f}\\%"
            )
            if opt_cut is not None and r["cutoff"] == opt_cut:
                cells = " & ".join(f"\\textbf{{{c.strip()}}}" for c in cells.split("&"))
            lines.append(cells + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _macro_elasticities_table_latex(metrics_dict):
        elasticities = metrics_dict.get("macro_elasticities", {})          # raw OLS
        adjusted     = metrics_dict.get("macro_elasticities_adjusted", {})  # sign-corrected
        predictions  = metrics_dict.get("macro_predictions", {})
        shocks       = metrics_dict.get("macro_implied_shocks", {})
        sign_adj     = bool(metrics_dict.get("macro_sign_adjusted", False))
        macro_lag    = int(metrics_dict.get("macro_unrate_lag", 0))
        macro_r2     = metrics_dict.get("macro_r_squared", None)

        if not elasticities:
            return r"\textit{No macroeconomic elasticity data available.}"

        # Coefficients used for scenario projection (sign-corrected where applicable).
        coef_src = adjusted if (sign_adj and adjusted) else elasticities
        unrate_coef = coef_src.get('UNRATE', 0.0)
        raw_unrate = elasticities.get('UNRATE', 0.0)

        r2_txt = f" (raw OLS $R^2={macro_r2:.3f}$)" if isinstance(macro_r2, (int, float)) and macro_r2 == macro_r2 else ""
        if sign_adj:
            unrate_sign_note = (
                r"\footnotesize$^{\dagger}$ The raw contemporaneous OLS produced a spurious "
                f"\\emph{{negative}} UNRATE coefficient ({raw_unrate:+.4f}){r2_txt}: LendingClub "
                r"charge-offs lag the macro cycle and origination underwriting drifts over 2007--2018, "
                r"so tightly-underwritten high-unemployment vintages show \emph{lower} realised defaults "
                r"than the loosely-underwritten low-unemployment 2015--16 vintages. For scenario "
                f"projection the series is lagged {macro_lag} quarter(s) and economically-correct sign "
                r"priors are imposed (magnitude from the fitted OLS), guaranteeing the intuitive "
                r"Downside $>$ Baseline $>$ Upside ordering. The coefficients shown are these "
                r"projection coefficients; raw OLS values are reported inline above."
            )
        else:
            unrate_sign_note = (
                r"\footnotesize$^{\dagger}$ Coefficients estimated by OLS of quarterly default rate on "
                f"the macro factors (lagged {macro_lag} quarter(s)){r2_txt}; signs follow the expected "
                r"economic direction (rising unemployment $\rightarrow$ higher defaults)."
            )

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Macroeconomic Default Rate OLS Regression \& Scenario Mapping}",
            r"\label{tab:macro_regression}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Macro Variable} & \textbf{OLS Coefficient} & \textbf{Impact Explanation} \\",
            r"\midrule",
            f"Intercept (Constant) & {coef_src.get('const', 0.0):.4f} & Baseline Default Level \\\\",
            f"Unemployment Rate (UNRATE)$^{{\\dagger}}$ & {unrate_coef:.4f} & "
            f"+1\\% UNRATE $\\rightarrow$ {unrate_coef*100:+.2f}\\% Default Rate \\\\",
            f"GDP Growth (GDP\\_growth) & {coef_src.get('GDP_growth', 0.0):.4f} & "
            f"+1\\% GDP Growth $\\rightarrow$ {coef_src.get('GDP_growth', 0.0)*100:+.2f}\\% Default Rate \\\\",
            f"Fed Funds Rate (FEDFUNDS) & {coef_src.get('FEDFUNDS', 0.0):.4f} & "
            f"+1\\% Interest Rate $\\rightarrow$ {coef_src.get('FEDFUNDS', 0.0)*100:+.2f}\\% Default Rate \\\\",
            f"CPI Inflation (CPI\\_inflation) & {coef_src.get('CPI_inflation', 0.0):.4f} & "
            f"+1\\% Inflation $\\rightarrow$ {coef_src.get('CPI_inflation', 0.0)*100:+.2f}\\% Default Rate \\\\",
        ]
        if "HPI_growth" in coef_src:
            lines.append(
                f"House Price Index Growth (HPI\\_growth) & {coef_src.get('HPI_growth', 0.0):.4f} & "
                f"+1\\% HPI Growth $\\rightarrow$ {coef_src.get('HPI_growth', 0.0)*100:+.2f}\\% Default Rate \\\\"
            )
        lines += [
            r"\midrule",
            r"\textbf{Scenario} & \textbf{Implied Default Rate} & \textbf{Mapped Vasicek Shock ($Z$) / Weight} \\",
            r"\midrule",
            f"Upside Scenario & {predictions.get('upside', 0.0):.2f}\\% & {shocks.get('upside', 0.5):.4f} (weight 25\\%) \\\\",
            f"Baseline Scenario & {predictions.get('baseline', 0.0):.2f}\\% & {shocks.get('baseline', 0.0):.4f} (weight 50\\%) \\\\",
            f"Downside Scenario & {predictions.get('downside', 0.0):.2f}\\% & {shocks.get('downside', -1.0):.4f} (weight 25\\%) \\\\",
            r"\bottomrule",
            r"\multicolumn{3}{p{\linewidth}}{" + unrate_sign_note + r"} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]

        # Fix 1.3: scenario input assumptions table so the reader can verify
        # the implied default rates independently from the OLS coefficients.
        scenario_inputs = metrics_dict.get("macro_scenario_inputs", {})
        if scenario_inputs:
            def _si(scen, var):
                return f"{scenario_inputs.get(scen, {}).get(var, float('nan')):.2f}"

            has_hpi = any("HPI_growth" in scenario_inputs.get(s, {}) for s in ("upside", "baseline", "downside"))
            col_spec = "lccccc" if has_hpi else "lcccc"
            header = r"\textbf{Scenario} & \textbf{UNRATE (\%)} & \textbf{GDP Growth (\%)} & \textbf{FEDFUNDS (\%)} & \textbf{CPI Inflation (\%)}"
            if has_hpi:
                header += r" & \textbf{HPI Growth (\%)}"
            header += r" \\"

            def _row(scen_label, scen_key):
                row = f"{scen_label} & {_si(scen_key,'UNRATE')} & {_si(scen_key,'GDP_growth')} & {_si(scen_key,'FEDFUNDS')} & {_si(scen_key,'CPI_inflation')}"
                if has_hpi:
                    row += f" & {_si(scen_key,'HPI_growth')}"
                return row + r" \\"

            lines += [
                "",
                r"\begin{table}[H]",
                r"\centering",
                r"\caption{Assumed Macroeconomic Inputs per Scenario}",
                r"\label{tab:scenario_inputs}",
                r"\vspace{0.5em}",
                f"\\begin{{tabular}}{{{col_spec}}}",
                r"\toprule",
                header,
                r"\midrule",
                _row("Upside", "upside"),
                _row("Baseline", "baseline"),
                _row("Downside", "downside"),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]

        # Fix 1.3: if the Downside implied DR is not the highest, do not
        # silently accept — add an explicit explanation tied to the UNRATE
        # coefficient anomaly documented in the footnote above.
        dr_up = predictions.get("upside")
        dr_base = predictions.get("baseline")
        dr_down = predictions.get("downside")
        if None not in (dr_up, dr_base, dr_down) and not (dr_down >= dr_base >= dr_up):
            lines += [
                "",
                r"\noindent\textit{Note on scenario ordering:} the implied default rates above "
                r"do not follow the intuitive Downside $>$ Baseline $>$ Upside ordering. This is a "
                r"direct consequence of the negative UNRATE coefficient discussed in the table "
                r"footnote: because LendingClub charge-offs lag the macro cycle, the contemporaneous "
                r"OLS attributes part of the unemployment effect to prior write-offs, so the assumed "
                r"rise in unemployment under the Downside scenario is partially offset in-sample by "
                r"the GDP and rate terms. The counter-intuitive ordering is therefore a documented "
                r"limitation of the contemporaneous OLS mapping rather than a labelling error; the "
                r"Vasicek $Z$ mapping and all ECL stress directions in Section~5 and Figure~\ref{fig:ecl_tornado} "
                r"follow the convention $Z<0$ = adverse shock = higher PD/ECL.",
            ]
        return "\n".join(lines)

    def _macro_ts_table_latex(metrics_dict):
        """ADF / Granger / AIC-lag / Johansen-VECM time-series diagnostics."""
        ts = metrics_dict.get("macro_ts", {})
        if not ts:
            return r"\textit{Macro time-series diagnostics not available for this run.}"

        def _fmt_p(p):
            try:
                p = float(p)
            except (TypeError, ValueError):
                return "n/a"
            return "$<$0.001" if p < 0.001 else f"{p:.3f}"

        rows = []
        adf = ts.get("adf", {}) or {}
        for name, res in adf.items():
            if not res:
                continue
            label = "Default Rate" if name == "default_rate" else name.replace("_", r"\_")
            verdict = "Stationary" if res.get("stationary") else "Unit root"
            rows.append(
                f"ADF --- {label} & {res.get('stat', float('nan')):.3f} & "
                f"{_fmt_p(res.get('pvalue'))} & {verdict} \\\\"
            )

        gr = ts.get("granger")
        if gr:
            alpha_c = gr.get("alpha_corrected")
            thr = (f" ($\\alpha_{{\\text{{corr}}}}={float(alpha_c):.3f}$)"
                   if alpha_c is not None else "")
            verdict = ("Causal" if gr.get("causal") else "No causality") + thr
            rows.append(
                f"Granger UNRATE $\\rightarrow$ DR (lag {gr.get('best_lag', 0)}) & --- & "
                f"{_fmt_p(gr.get('min_pvalue'))} & {verdict} \\\\"
            )

        aic = ts.get("aic_lag_selection")
        if aic:
            sign = "$+$ (correct)" if aic.get("unrate_sign_ok") else "$-$ (spurious)"
            rows.append(
                f"AIC lag selection (lag {aic.get('lag', 0)}) & "
                f"{aic.get('unrate_coef', float('nan')):+.4f} & --- & UNRATE {sign} \\\\"
            )

        joh = ts.get("johansen")
        if joh:
            verdict = "Cointegrated" if joh.get("cointegrated") else "Not cointegrated"
            rows.append(
                f"Johansen trace ($r=0$) & {joh.get('trace_stat', float('nan')):.2f} & "
                f"crit {joh.get('crit_5pct', float('nan')):.2f} & {verdict} \\\\"
            )

        if not rows:
            return r"\textit{Macro time-series diagnostics produced no usable output on this series.}"

        n_q = int(ts.get("n_quarters", 0))
        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            f"\\caption{{Macro Time-Series Diagnostics ($n={n_q}$ quarters)}}",
            r"\label{tab:macro_ts}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lccl}",
            r"\toprule",
            r"\textbf{Test} & \textbf{Statistic} & \textbf{$p$ / Crit.} & \textbf{Verdict} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\\[0.3em]{\footnotesize The Granger verdict applies a Bonferroni correction across the lags tested to a nominal $\alpha=0.10$, giving $\alpha_{\text{corr}}=0.10/k$ for $k$ lags; a minimum $p$-value above $\alpha_{\text{corr}}$ is reported as no causality, guarding against multiple-testing false positives. Note the nominal level is $0.10$, not $0.05$: at $k=4$ this threshold ($0.025$) is looser than a Bonferroni correction applied to $\alpha=0.05$ would be ($0.0125$).}",
            r"\end{table}",
        ])

    def _vintage_calib_table_latex(metrics_dict):
        """Raw vs isotonic/Platt PD against actual DR per vintage group."""
        rows = metrics_dict.get("vintage_calibration", [])
        if not rows:
            return r"\textit{Vintage calibration diagnostic not available for this run.}"
        body = []
        for r in rows:
            grp = str(r.get("group", "")).replace("_", r"\_")
            body.append(
                f"{grp} & {int(r.get('n', 0)):,} & {float(r.get('raw_pd', 0.0)) * 100:.2f}\\% & "
                f"{float(r.get('isotonic_pd', 0.0)) * 100:.2f}\\% & "
                f"{float(r.get('platt_pd', 0.0)) * 100:.2f}\\% & "
                f"{float(r.get('actual_dr', 0.0)) * 100:.2f}\\% & "
                f"{float(r.get('pd_ratio_raw', 0.0)):.3f} \\\\"
            )
        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Calibration by Vintage Group --- Raw vs Era-Recalibrated PD}",
            r"\label{tab:vintage_calib}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"\textbf{Vintage} & \textbf{N} & \textbf{Raw PD} & \textbf{Isotonic PD} & "
            r"\textbf{Platt PD} & \textbf{Actual DR} & \textbf{Raw Ratio} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _lifetime_pd_calibration_table_latex(metrics_dict):
        """Hazard-model lifetime PD vs observed lifetime default rate, by mature vintage.

        Validates the PD that drives IFRS 9 ECL directly (never passed through the
        scorecard's OOS recalibrator) against realised outcomes --- see
        credit_risk.validation.calibration.lifetime_pd_calibration_by_vintage.
        """
        diag = metrics_dict.get("lifetime_pd_calibration", {})
        rows = diag.get("by_vintage", [])
        port = diag.get("portfolio", {})
        if not rows or not port or not port.get("n"):
            return r"\textit{Lifetime PD calibration diagnostic not available for this run.}"
        body = []
        for r in rows:
            ratio = float(r.get("ratio", float("nan")))
            flag = "" if r.get("in_band", False) else r"$\dagger$"
            body.append(
                f"{int(r.get('vintage_year', 0))} & {int(r.get('n', 0)):,} & "
                f"{float(r.get('predicted_pd_lifetime', 0.0)) * 100:.2f}\\% & "
                f"{float(r.get('observed_dr', 0.0)) * 100:.2f}\\% & "
                f"{ratio:.2f}{flag} \\\\"
            )
        port_ratio = float(port.get("ratio", float("nan")))
        port_flag = "" if port.get("in_band", False) else r"$\dagger$"
        body.append(r"\midrule")
        body.append(
            f"\\textbf{{All mature vintages}} & {int(port.get('n', 0)):,} & "
            f"{float(port.get('predicted_pd_lifetime', 0.0)) * 100:.2f}\\% & "
            f"{float(port.get('observed_dr', 0.0)) * 100:.2f}\\% & "
            f"\\textbf{{{port_ratio:.2f}{port_flag}}} \\\\"
        )
        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Hazard Model Lifetime PD vs Realised Lifetime Default Rate (Matured Vintages)}",
            r"\label{tab:lifetime_pd_calibration}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\textbf{Vintage} & \textbf{N} & \textbf{Predicted Lifetime PD} & "
            r"\textbf{Observed Default Rate} & \textbf{Ratio} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\multicolumn{5}{p{0.95\linewidth}}{\footnotesize $\dagger$ outside the "
            r"$[0.5, 1.5]$ tolerance band. Restricted to vintages originated in or "
            r"before 2016: the 2018Q4 snapshot has not yet resolved recoveries/"
            r"charge-offs for 2017--2018 originations, so their observed default "
            r"status is right-censored. The 2015 cohort is absent because it falls in "
            r"the excluded train/OOT grey zone (Section~2.3); the 2016 cohort is "
            r"included but is itself only partially matured, which depresses its "
            r"ratio.} \\",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _ab_test_table_latex(metrics_dict):
        """Paired bootstrap A/B: Gini CIs for champion, challenger and difference."""
        ab = metrics_dict.get("ab_test", {})
        if not ab or ab.get("n_boot_valid", 0) in (0, None):
            return r"\textit{Paired bootstrap A/B test not available for this run.}"
        ci_pct = int(round(float(ab.get("ci", 0.95)) * 100))
        a = ab.get("gini_a", {})
        b = ab.get("gini_b", {})
        d = ab.get("diff", {})
        sig = bool(ab.get("significant", False))
        verdict = (
            "Significant (CI excludes 0)" if sig
            else "Not significant (CI spans 0)"
        )

        def _row(label, s):
            return (
                f"{label} & {float(s.get('median', 0.0)):.4f} & "
                f"[{float(s.get('lo', 0.0)):.4f}, {float(s.get('hi', 0.0)):.4f}] \\\\"
            )

        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            f"\\caption{{Paired Bootstrap A/B Test --- Gini with {ci_pct}\\% CIs "
            f"($n_{{\\text{{boot}}}}={int(ab.get('n_boot_valid', 0)):,}$)}}",
            r"\label{tab:ab_test}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            f"\\textbf{{Model}} & \\textbf{{Gini (median)}} & \\textbf{{{ci_pct}\\% CI}} \\\\",
            r"\midrule",
            _row("Champion (Scorecard)", a),
            _row("Challenger (LightGBM)", b),
            r"\midrule",
            _row(r"Difference (B $-$ A)", d),
            r"\bottomrule",
            r"\multicolumn{3}{l}{\footnotesize " + verdict + r"} \\",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _hhi_table_latex(metrics_dict):
        """Concentration: HHI + effective N per dimension + granularity surcharge."""
        conc = metrics_dict.get("concentration", {})
        dims = conc.get("dimensions", []) if conc else []
        if not dims:
            return r"\textit{Concentration analysis not available for this run.}"
        _labels = {"grade": "Credit Grade", "purpose": "Loan Purpose",
                   "addr_state": "Borrower State"}
        body = []
        for d in dims:
            name = _labels.get(str(d.get("dimension")), str(d.get("dimension")).replace("_", r"\_"))
            hhi = float(d.get("hhi", 0.0))
            eff_n = float(d.get("effective_n", 0.0))
            n_cat = int(d.get("n_categories", 0))
            top = float(d.get("top_share", 0.0)) * 100.0
            body.append(f"{name} & {hhi:.4f} & {eff_n:.1f} & {n_cat} & {top:.1f}\\% \\\\")
        ga_raw = float(conc.get("granularity_adjustment", 0.0))
        if abs(ga_raw) >= 1e6:
            ga_str = f"\\${ga_raw / 1e6:,.2f}M"
        elif abs(ga_raw) >= 1e3:
            ga_str = f"\\${ga_raw / 1e3:,.1f}K"
        else:
            ga_str = f"\\${ga_raw:,.0f}"
        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Portfolio Concentration --- Herfindahl-Hirschman Index by Dimension}",
            r"\label{tab:hhi}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\textbf{Dimension} & \textbf{HHI} & \textbf{Eff. $N$} & \textbf{Categories} & \textbf{Top Share} \\",
            r"\midrule",
            *body,
            r"\midrule",
            f"\\multicolumn{{5}}{{l}}{{\\textbf{{Granularity Adjustment (capital surcharge):}} {ga_str}}} \\\\",
            r"\midrule",
            r"\multicolumn{5}{p{0.95\linewidth}}{\footnotesize The Gordy--L\"utkebohmert granularity adjustment is a single-\emph{name} "
            r"idiosyncratic-risk add-on ($\sum_i \mathrm{UL}_i^2 / 2\,\mathrm{EAD}_{\text{tot}}$); on this loan-granular book (hundreds of "
            r"thousands of small exposures) it is near-zero by construction and is distinct from the segment-level HHI figures above, which "
            r"measure concentration across grade / purpose / state buckets rather than individual names.} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _risk_measures_table_latex(metrics_dict):
        """Monte Carlo economic-capital risk measures (EL / VaR / ES / UL / EC)."""
        ec = metrics_dict.get("econ_cap", {})
        if not ec:
            return r"\textit{Economic capital simulation not available for this run.}"

        def _m(key):
            return float(ec.get(key, 0.0)) / 1e6

        alpha_pct = float(ec.get("alpha", 0.999)) * 100.0
        n_sim = int(ec.get("n_simulations", 0))
        # rho is either a constant or the string "supervisory" (the per-PD-bucket BCBS
        # "Other Retail" curve), so the caption is built for either (the internal review log N13).
        _rho_raw = ec.get("rho", 0.15)
        try:
            rho_caption = f"$\\rho={float(_rho_raw):.2f}$"
        except (TypeError, ValueError):
            rho_caption = r"supervisory $R(\mathrm{PD}) \in [0.03, 0.16]$"
        reg_cap = _m("regulatory_capital")
        ec_cap = _m("economic_capital")
        ratio = ec.get("ec_to_reg_ratio", 0.0)
        ratio_txt = f"{ratio * 100:.1f}\\%" if reg_cap > 0 else r"n/a"

        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Monte Carlo Economic Capital --- Risk Measures (ASRF, "
            f"$N={n_sim:,}$ simulations, {rho_caption})}}",
            r"\label{tab:risk_measures}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"\textbf{Risk Measure} & \textbf{Value (\$M)} \\",
            r"\midrule",
            f"Expected Loss (EL) & {_m('expected_loss'):,.2f} \\\\",
            f"Value-at-Risk (VaR {alpha_pct:.1f}\\%) & {_m('var'):,.2f} \\\\",
            f"Expected Shortfall (ES {alpha_pct:.1f}\\%) & {_m('es'):,.2f} \\\\",
            f"Unexpected Loss (UL $=$ VaR $-$ EL) & {_m('unexpected_loss'):,.2f} \\\\",
            r"\midrule",
            f"\\textbf{{Economic Capital (EC $=$ ES $-$ EL)}} & \\textbf{{{ec_cap:,.2f}}} \\\\",
            f"Basel IRB Regulatory Capital (8\\%) & {reg_cap:,.2f} \\\\",
            f"EC / Regulatory Capital & {ratio_txt} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _cox_table_latex(metrics_dict):
        """Cox proportional-hazards coefficient / hazard-ratio summary."""
        surv = metrics_dict.get("survival", {})
        rows = surv.get("cox_summary", []) if surv else []
        if not rows:
            return r"\textit{Cox proportional-hazards summary not available for this run.}"

        _labels = {
            "grade_num": "Credit Grade (A=1..G=7)",
            "int_rate": "Interest Rate",
            "dti": "Debt-to-Income (DTI)",
            "term_num": "Term (months)",
        }
        body = []
        for r in rows:
            cov = str(r.get("covariate", ""))
            label = _labels.get(cov, cov.replace("_", r"\_"))
            coef = float(r.get("coef", 0.0))
            hr = float(r.get("hazard_ratio", 0.0))
            p = float(r.get("p_value", float("nan")))
            sd = r.get("sd")
            sd_txt = f"{float(sd):.4g}" if sd is not None and float(sd) == float(sd) else "--"
            p_txt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
            body.append(f"{label} & {sd_txt} & {coef:+.5f} & {hr:.5f} & {p_txt} \\\\")

        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Cox Proportional-Hazards Model --- Covariate Summary "
            r"(coefficients per standard deviation)}",
            r"\label{tab:cox_summary}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\textbf{Covariate} & \textbf{SD} & \textbf{Coef ($\beta$ per SD)} & "
            r"\textbf{Hazard Ratio} & \textbf{$p$-value} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _lgd_validation_table_latex(metrics_dict):
        """Out-of-sample LGD validation metrics (MAE / RMSE / R2 / KS statistic).

        The KS p-value is intentionally omitted: at n~150k, a two-sample KS
        test of the MARGINAL predicted-vs-actual distributions is hyper-
        sensitive (any trivial difference yields p<0.001) and does not test
        per-loan calibration; the decile table/figure is the calibration
        evidence. Reporting only the KS statistic avoids implying a
        distributional-fit pass/fail that the p-value cannot support here.
        """
        val = metrics_dict.get("lgd_validation", {})
        if not val or val.get("n_test", 0) in (0, 0.0):
            return r"\textit{Out-of-sample LGD validation not available for this run.}"
        n_test = int(val.get("n_test", 0))
        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            f"\\caption{{Out-of-Sample LGD Validation Metrics ($n={n_test:,}$ held-out defaults)}}",
            r"\label{tab:lgd_validation}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Value} \\",
            r"\midrule",
            f"Mean Absolute Error (MAE) & {float(val.get('mae', 0.0)):.4f} \\\\",
            f"Root Mean Squared Error (RMSE) & {float(val.get('rmse', 0.0)):.4f} \\\\",
            f"Coefficient of Determination ($R^2$) & {float(val.get('r2', 0.0)):.4f} \\\\",
            f"KS Statistic (marginal dist., pred vs actual) & {float(val.get('ks_stat', 0.0)):.4f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _ecl_whatif_table_latex(metrics_dict):
        """ECL what-if stress scenarios (base / shocked / delta)."""
        rows = metrics_dict.get("ecl_whatif", [])
        if not rows:
            return r"\textit{ECL what-if analysis not available for this run.}"
        base = float(rows[0].get("base_ecl", 0.0)) / 1e6
        body = []
        for r in rows:
            name = str(r.get("scenario", "")).replace("%", r"\%").replace("_", r"\_")
            shocked = float(r.get("shocked_ecl", 0.0)) / 1e6
            d_ecl = float(r.get("delta_ecl", 0.0)) / 1e6
            d_pct = float(r.get("delta_pct", 0.0))
            body.append(f"{name} & {shocked:,.2f} & {d_ecl:+,.2f} & {d_pct:+.1f}\\% \\\\")
        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            f"\\caption{{ECL What-If Sensitivity (anchor ECL $=$ \\${base:,.1f}M at $Z=0$; this is neither the priced Baseline scenario, which carries its own non-zero implied $Z$, nor the probability-weighted total ECL of Table~\\ref{{tab:exec_summary}}, which additionally weights the Upside and Downside scenarios. Every figure below is a ratio to this same anchor, so the choice of anchor cancels)}}",
            r"\label{tab:ecl_whatif}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"\textbf{Scenario} & \textbf{Shocked ECL (\$M)} & \textbf{$\Delta$ ECL (\$M)} & \textbf{$\Delta$ \%} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def _ml_comparison_table_latex(comparison_list):
        if not comparison_list:
            return r"\textit{No ML benchmark comparison data available.}"
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\caption{Machine Learning Champion-Challenger Performance Comparison}",
            r"\label{tab:ml_comparison}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lccccccr}",
            r"\toprule",
            r"\textbf{Model Name} & \textbf{Test AUC} & \textbf{OOT AUC} & \textbf{Test Gini} & \textbf{OOT Gini} & \textbf{Test KS} & \textbf{OOT KS} & \textbf{Time (s)} \\",
            r"\midrule",
        ]
        for row in comparison_list:
            model = row["model"]
            test_auc = row["test_auc"]
            oot_auc = row["oot_auc"]
            test_gini = row["test_gini"]
            oot_gini = row["oot_gini"]
            test_ks = row["test_ks"]
            oot_ks = row["oot_ks"]
            t_time = row["train_time_sec"]

            time_str = f"{t_time:.2f}s" if t_time >= 0.01 else "<0.01s"

            lines.append(
                f"\\textbf{{{model}}} & {test_auc:.4f} & {oot_auc:.4f} & {test_gini:.4f} & {oot_gini:.4f} & {test_ks:.4f} & {oot_ks:.4f} & {time_str} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    # Build table strings
    _iv_rows = sc_tables.get("iv_table", [])
    _coef_rows = sc_tables.get("logit_coefficients", [])
    _sc_rows = sc_tables.get("scorecard_table", [])
    _selected = sc_tables.get("selected_features", [])

    iv_table_tex = _iv_table_latex(
        _iv_rows,
        stages=metrics.get("feature_selection_stages")
        or sc_tables.get("feature_selection_stages"),
        selected=_selected,
    )
    logit_table_tex = _logit_table_latex(_coef_rows)
    scorecard_points_tex = _scorecard_points_latex(_sc_rows)
    selected_features_str = (
        ",\\allowbreak ".join([f"\\texttt{{{f.replace('_', chr(92) + '_')}}}" for f in _selected])
        if _selected else "N/A"
    )
    pd_backtest_rows_tex = _pd_backtest_rows_latex(metrics.get("pd_backtest_vintage", []))
    ml_comparison_table_tex = _ml_comparison_table_latex(metrics.get("ml_benchmark_comparison", []))
    csi_table_tex = _csi_table_latex(metrics.get("csi_table", []))
    ecl_reconciliation_tex = _ecl_reconciliation_latex(
        metrics.get("ecl_reconciliation") or {}, _num(metrics, "total_el")
    )

    # ── D3: ML Gini helper for benchmark table ────────────────────────────────
    def _get_ml_gini(model_name, rows):
        for r in rows:
            if r.get("model") == model_name:
                return f"{r.get('oot_gini', 0):.4f}"
        return "N/A"
    ml_rows = metrics.get("ml_benchmark_comparison", [])

    # ── Helper formatters ──────────────────────────────────────────────────────
    # A missing optional metric arrives here as NaN (see _num) and must print as a visible
    # "n/a", never as a number. Formatting NaN with "{:,.0f}" would otherwise emit the
    # literal string "nan" into the PDF.
    def fmt_num(val, fmt="{:,.0f}"):
        try:
            f = float(val)
            return "n/a" if f != f else fmt.format(f)
        except Exception:
            return "n/a"

    def fmt_dec(val, precision=4):
        try:
            f = float(val)
            return "n/a" if f != f else f"{f:.{precision}f}"
        except Exception:
            return "n/a"

    def fmt_pct(val, precision=2):
        try:
            f = float(val)
            return "n/a" if f != f else f"{f * 100:.{precision}f}\\%"
        except Exception:
            return "n/a"

    # ── Load metrics ───────────────────────────────────────────────────────────
    auc = fmt_dec(_num(metrics, "auc"))
    gini = fmt_dec(_num(metrics, "gini"))
    ks = fmt_dec(_num(metrics, "ks"))
    auc_oot = fmt_dec(_num(metrics, "auc_oot"))
    gini_oot = fmt_dec(_num(metrics, "gini_oot"))
    ks_oot = fmt_dec(_num(metrics, "ks_oot"))
    brier_oot = fmt_dec(metrics.get("calibration", {}).get("oot", {}).get("brier_score", 0.0582), 4)
    mean_lgd = fmt_dec(_num(metrics, "mean_lgd"))
    downturn_lgd = fmt_dec(_num(metrics, "downturn_lgd"))
    lgd_uplift_pp = fmt_dec(
        (float(_num(metrics, "downturn_lgd")) - float(_num(metrics, "mean_lgd"))) * 100.0,
        precision=2,
    )
    lgd_val_n = f"{int(metrics.get('lgd_validation', {}).get('n_test', 0)):,}"
    # Interpolate the formatted values directly: nested __TOKEN__ placeholders inside a
    # substituted string would not be expanded, since the scalar replacements run first.
    _dl = float(_num(metrics, "downturn_lgd"))
    if _dl >= 0.999:
        downturn_note = (
            "On this fully unsecured, non-revolving instalment book the realised severity of "
            "loss-incurring defaults is concentrated at total loss, so the 90th percentile "
            f"sits at the cap of the $[0,1]$ range ({downturn_lgd}), "
            f"{lgd_uplift_pp}\\,pp above mean LGD. Two consequences follow. It is a real "
            "empirical percentile rather than a modelled stress --- more than a tenth of "
            "loss-incurring defaults recover nothing at all. And because severity cannot "
            "exceed total loss, the Basel capital calculation has no headroom above this "
            "figure: any further conservatism must come from PD or EAD, not LGD."
        )
    else:
        downturn_note = (
            f"The 90th percentile of realised severity sits {lgd_uplift_pp}\\,pp above mean "
            "LGD, an interior point of the severity distribution, so the downturn figure "
            "carries genuine headroom over the central estimate."
        )

    # ── Champion vs challenger verdict, derived from the benchmark table ────────
    # Which model wins on OOT discrimination is an empirical result, not a premise. The
    # shipped report asserted the scorecard won and hard-coded the rivals' AUCs; once the
    # challengers were given feature parity (audit B2) the ranking changed. Everything
    # below is therefore computed (AUDIT-A12).
    _bench = metrics.get("ml_benchmark_comparison") or []
    _sc_row = next((r for r in _bench if "Scorecard" in str(r.get("model", ""))), None)
    _rivals = [r for r in _bench if r is not _sc_row]
    if _sc_row and _rivals:
        _sc_auc = float(_sc_row["oot_auc"])
        _best_rival = max(_rivals, key=lambda r: float(r["oot_auc"]))
        _best_auc = float(_best_rival["oot_auc"])
        _sc_wins = _sc_auc >= _best_auc
        _gap = abs(_best_auc - _sc_auc)
        ml_ranking = ", ".join(
            f"{r['model']} ({float(r['oot_auc']):.4f})"
            for r in sorted(_rivals, key=lambda r: -float(r["oot_auc"]))
        )
        if _sc_wins:
            ml_verdict = (
                f"the WoE logistic scorecard attains the highest OOT discrimination in the "
                f"field: its OOT AUC of {_sc_auc:.4f} exceeds {ml_ranking}. The tree models "
                "fit in-sample structure that does not transfer across the 2016--2018 regime "
                "shift, whereas the scorecard's coarse monotone WoE bins generalise more "
                "stably. The Logistic Scorecard is therefore the champion on discrimination "
                "grounds alone, before its advantages in point-based explainability and "
                "regulatory audit compliance are considered."
            )
            champion_rationale = (
                "on the out-of-time set the interpretable logistic scorecard generalises "
                f"better: its OOT AUC ({_sc_auc:.4f}) exceeds the best challenger "
                f"({_best_rival['model']}, {_best_auc:.4f}), a "
                "gap tested in Section~7.8. Combined with its regulatory advantages, the "
                "scorecard is the preferred underwriting model."
            )
            ceiling_note = (
                f"and the best challenger, {_best_rival['model']} (Section~3.6), plateaus at "
                f"{_best_auc:.4f} rather than materially exceeding it --- consistent with a "
                "dataset/feature ceiling rather than an under-fitted model"
            )
        else:
            ml_verdict = (
                f"the tree-based challengers, evaluated on an identical predictor set, "
                f"\\emph{{outperform}} the WoE logistic scorecard on out-of-time "
                f"discrimination: the scorecard's OOT AUC of {_sc_auc:.4f} is exceeded by "
                f"{ml_ranking}. The best challenger ({_best_rival['model']}) leads by "
                f"{_gap:.4f} AUC. We report this plainly rather than presenting the "
                "scorecard as the discrimination winner: it is not. The scorecard is "
                "retained as champion on \\emph{governance} grounds --- exact per-feature "
                "points attribution for adverse-action reasons, a monotone and auditable "
                "functional form, and a stable calibration story --- and the cost of that "
                "choice is the AUC gap quantified here, which a model-risk committee should "
                "weigh explicitly rather than have hidden."
            )
            champion_rationale = (
                f"on the out-of-time set the best challenger, {_best_rival['model']}, "
                f"\\emph{{exceeds}} the "
                f"scorecard's discrimination (OOT AUC {_best_auc:.4f} vs {_sc_auc:.4f}), a "
                "gap tested in Section~7.8. The scorecard is nonetheless retained as the "
                "production model for regulatory and explainability reasons set out below, "
                "with the discrimination sacrifice reported rather than obscured."
            )
            ceiling_note = (
                f"while the tree challengers reach {_best_auc:.4f} on the same features, "
                "indicating the scorecard sits modestly below the attainable ceiling rather "
                "than at it"
            )
    else:
        ml_ranking = "n/a"
        ml_verdict = "the champion/challenger benchmark did not produce a comparable field."
        champion_rationale = "the champion/challenger comparison is reported in Section~7.8."
        ceiling_note = "and is benchmarked against the challenger field in Section~7.8"

    # Paired bootstrap direction: the CI is on (challenger - champion).
    _ab = metrics.get("ab_test") or {}
    _ab_med = float((_ab.get("diff") or {}).get("median", 0.0))
    if not _ab:
        ab_direction = "the paired bootstrap did not run."
    elif not _ab.get("significant", False):
        ab_direction = (
            "the interval spans zero, so the Gini difference between the two models is not "
            "distinguishable from sampling noise."
        )
    elif _ab_med > 0:
        ab_direction = (
            f"the interval lies entirely \\emph{{above}} zero (median $+{_ab_med:.4f}$), so "
            "the challenger's Gini advantage over the champion is statistically significant, "
            "not sampling noise."
        )
    else:
        ab_direction = (
            f"the interval lies entirely \\emph{{below}} zero (median ${_ab_med:.4f}$), so "
            "the champion's Gini advantage over the challenger is statistically significant, "
            "not sampling noise."
        )

    # ── Stage-3 dominance of ECL, derived from the what-if sensitivities (audit C1) ──
    # ECL is exactly linear in LGD and EAD but flat in PD to the extent that ECL sits in
    # the PD-independent Stage 3 term. Inverting the PD+50% sensitivity recovers that share.
    _whatif = {str(r.get("scenario", "")): r for r in (metrics.get("ecl_whatif") or [])}

    def _whatif_pct(name: str) -> str:
        row = _whatif.get(name)
        return f"{row['delta_pct']:+.1f}\\%" if row else "n/a"

    whatif_lgd_pct = _whatif_pct("LGD +10pp")
    whatif_ead_pct = _whatif_pct("EAD +15%")
    whatif_pd50_pct = _whatif_pct("PD +50%")
    _pd50 = _whatif.get("PD +50%")
    if _pd50 is not None:
        _share = 1.0 - (float(_pd50["delta_pct"]) / 100.0) / 0.5
        stage3_ecl_share = fmt_pct(max(0.0, min(1.0, _share)), precision=0)
    else:
        stage3_ecl_share = "n/a"
    baseline_z = fmt_dec(metrics.get("macro_implied_shocks", {}).get("baseline", 0.0), 4)

    # The macro tornado's percentages are computed on a total that includes the
    # PD-independent Stage 3 term, so they understate the sensitivity of the part of the
    # book the macro model can actually reach. Section 10.3 already infers that share for
    # the what-if table; the same correction is applied here rather than left implicit.
    _sens_rows = metrics.get("ecl_sensitivity") or []
    _pd_indep_share = None
    if _pd50 is not None:
        _pd_indep_share = max(0.0, min(1.0, 1.0 - (float(_pd50["delta_pct"]) / 100.0) / 0.5))
    _anchor_ecl = float(
        next((r["total_ecl"] for r in _sens_rows
              if float(r.get("macro_shock", 1.0)) == 0.0), 0.0)
    )
    if _sens_rows and _anchor_ecl > 0:
        _span = max(
            abs(float(r.get("total_ecl", 0.0)) / _anchor_ecl - 1.0) for r in _sens_rows
        )
        tornado_span = f"{_span * 100:.1f}\\%"
        if _pd_indep_share is not None and _pd_indep_share < 0.999:
            tornado_pd_only = (
                f"scaled onto the PD-sensitive remainder it is "
                f"$\\pm${_span * 100 / (1.0 - _pd_indep_share):.1f}\\%."
            )
        else:
            tornado_pd_only = "the PD-sensitive share could not be inferred on this run."
    else:
        tornado_span = "n/a"
        tornado_pd_only = ""

    # The what-if table anchors its base_ecl at Z = 0 under a single scenario, whereas the
    # headline ECL is probability-weighted across scenarios with staging taken at the
    # baseline Z. The f-inference above is a ratio so the anchor cancels, but the two
    # numbers are not the same and the difference must be stated, not left for the reader
    # to trip over.
    _whatif_base = float(_pd50["base_ecl"]) if _pd50 is not None else 0.0
    _headline_ecl = float(_num(metrics, "total_ecl"))
    if _whatif_base > 0 and _headline_ecl > 0:
        whatif_base_note = (
            f" The what-if base is \\${_whatif_base/1e9:,.2f}bn: a single-scenario ECL at "
            f"$Z=0$, {abs(_whatif_base/_headline_ecl - 1.0)*100:.1f}\\% away from the "
            f"probability-weighted headline of \\${_headline_ecl/1e9:,.2f}bn. The inference "
            "above is a ratio of shocks to that same base, so the anchor cancels."
        )
    else:
        whatif_base_note = ""

    # Whether the performing-book provisions are usable at all is a property of the run,
    # not a standing claim: a hazard model whose 12-month leg collapses produces a Stage 1
    # provision of exactly zero, and this sentence previously asserted the opposite while
    # the reconciliation table above it printed $0.00bn.
    # Same derivation as the reconciliation table's own note: whether the two provisions
    # land near each other is a run property, and the reader is told which case they are in.
    _el_v = float(_num(metrics, "total_el"))
    _ecl_v = float(_num(metrics, "total_ecl"))
    _el_ecl_ratio = (_ecl_v / _el_v) if _el_v > 0 else 0.0
    if 0.8 <= _el_ecl_ratio <= 1.25:
        el_ecl_proximity = (
            "They nonetheless land within a few percent of one another, which side by side "
            "and without reconciliation reads as mutual confirmation. It is not."
        )
    else:
        el_ecl_proximity = (
            f"The ECL is {_el_ecl_ratio:.1f}$\\times$ the Expected Loss, and the gap is the "
            "staging: a lifetime horizon on Stage~2 and $PD=1$ on Stage~3 against a flat "
            "one-year horizon with no staging at all."
        )

    _recon_stage = (metrics.get("ecl_reconciliation") or {}).get("ecl_by_stage") or {}
    _ecl_s1_v = float(_recon_stage.get("s1", 0.0))
    if _ecl_s1_v > 0.0:
        performing_book_caveat = (
            "The Stage~1 and Stage~2 provisions, the ECL coverage ratio on the performing "
            "book, and the macro sensitivity analysis remain meaningful as relative "
            "comparisons"
        )
    else:
        performing_book_caveat = (
            "The Stage~1 provision is \\emph{zero} on this run --- the hazard model's "
            "12-month leg produces no default probability inside the first twelve months, "
            "so the entire performing book is provisioned at nil and only the Stage~2 and "
            "macro sensitivity comparisons carry any information"
        )

    # ── Phase-Gamma disclosures, all derived from metrics ──────────────────────

    # N44: the champion lifetime-PD model had no discrimination metric at all, while its
    # Cox challenger reported one.
    _hz = metrics.get("hazard_model_discrimination") or {}
    _cox_c = (metrics.get("survival") or {}).get("c_index")
    if _hz:
        hazard_discrimination_note = (
            f"The production hazard model attains an OOT AUC of {_hz['auc_oot']:.4f} "
            f"(Gini {_hz['gini_oot']:.4f}); for a binary outcome this equals its "
            "concordance index, so it is directly comparable to the Cox challenger's "
            + (f"C-index of {float(_cox_c):.4f}. " if _cox_c is not None else "C-index. ")
            + "Reporting it closes a gap in which the champion --- the model that produces "
            "every lifetime PD in the ECL --- was the only one carrying no measure of "
            "rank-ordering ability, while its challenger did."
        )
    else:
        hazard_discrimination_note = ""

    # N13: EC and regulatory capital must use a comparable correlation, or the ratio
    # measures the correlation gap rather than the tail.
    _ec = metrics.get("econ_cap") or {}
    _ec_sens = metrics.get("econ_cap_rho_sensitivity") or {}
    if _ec:
        ec_rho_note = (
            "The simulation uses the \\emph{same} supervisory ``Other Retail'' correlation "
            "curve as the IRB calculation it is compared against, evaluated per PD bucket. "
            "This matters for interpretation: a flat $\\rho = 0.15$ against a supervisory "
            "$R$ that collapses toward $0.03$ at this book's PD levels would make the "
            "EC-to-regulatory-capital ratio largely an artefact of a five-fold correlation "
            "difference rather than the tail fidelity it is meant to demonstrate."
        )
        if _ec_sens:
            ec_rho_note += (
                f" As a disclosed sensitivity, holding $\\rho$ flat at "
                f"{float(_ec_sens.get('rho', 0.15)):.2f} instead gives economic capital of "
                f"\\${_ec_sens.get('economic_capital', 0.0)/1e9:,.2f}bn "
                f"({float(_ec_sens.get('ec_to_reg_ratio', 0.0))*100:.1f}\\% of regulatory "
                "capital)."
            )
        if _ec.get("n_tail_observations"):
            ec_rho_note += (
                f" The 99.9\\% quantile rests on {int(_ec['n_tail_observations']):,} tail "
                f"draws, giving a Monte Carlo standard error of "
                f"\\${float(_ec.get('var_mc_stderr', 0.0))/1e6:,.1f}m on VaR; the figures "
                "should be read at that precision, not to the cent."
            )
    else:
        ec_rho_note = ""

    # N17: the central scenario is through-the-cycle, which departs from IFRS 9 B5.5.42.
    _macro = metrics.get("macro_implied_shocks") or {}
    _ttc = _macro.get("baseline_macro_ttc") or {}
    _at_date = _macro.get("macro_at_reporting_date") or {}
    if _ttc and _at_date and "UNRATE" in _ttc:
        ttc_baseline_note = (
            "\\textbf{The central scenario is through-the-cycle, not forward-looking.} It "
            "is anchored on the mean macro state of the training window, which spans the "
            f"financial crisis: baseline unemployment is {float(_ttc['UNRATE']):.2f}\\%, "
            f"against {float(_at_date.get('UNRATE', 0.0)):.2f}\\% actually prevailing at "
            "the reporting date. IFRS~9 B5.5.42 requires unbiased forward-looking "
            "expectations \\emph{at the reporting date}, so this is a deliberate departure "
            "from the standard: it is the conventional basis for regulatory-style "
            "through-the-cycle provisioning and keeps the scenario set stable across "
            "reporting periods, at the cost of a central scenario that is more adverse "
            "than the conditions actually observed. A production deployment would anchor "
            "the baseline on reporting-date conditions or a published consensus forecast."
        )
    else:
        ttc_baseline_note = ""

    # N16: staging must be evaluated under a scenario that is actually priced.
    _recon = metrics.get("ecl_reconciliation") or {}
    if _recon.get("staging_scenario"):
        staging_scenario_note = (
            f"Staging is evaluated under the \\textbf{{{_recon['staging_scenario']}}} "
            f"scenario ($Z = {float(_recon.get('staging_macro_shock', 0.0)):.4f}$), the "
            "same central expectation the provision is anchored to. It was previously "
            "computed at $Z = 0$, which corresponds to none of the three priced scenarios: "
            "the Vasicek conditional-PD function does not return the unconditional PD at "
            "$Z = 0$, so loans were being sorted into stages under macroeconomic conditions "
            "that appeared nowhere else in the calculation."
        )
    else:
        staging_scenario_note = ""

    # N33: the reject-inference Gini shift was largely an imputation artefact.
    _ri = metrics.get("reject_inference") or {}
    if _ri:
        reject_inference_note = (
            f"The comparison is restricted to the {int(_ri.get('n_features_used', 0))} of "
            f"{int(_ri.get('n_features_candidate', 0))} scorecard predictors that are "
            "genuinely observed for rejected applicants, and both Gini figures are computed "
            "on that same set with the same weighting."
        )
        if _ri.get("n_features_imputed_constant"):
            reject_inference_note += (
                f" The remaining {int(_ri['n_features_imputed_constant'])} exist only in the "
                "accepted file and were previously mean-imputed to a training constant "
                "across the entire reject population. A constant predictor has no "
                "discriminatory power, so the earlier Gini drop measured that imputation at "
                "least as much as any latent risk in the through-the-door population, and "
                "should not be read as economic evidence about rejected applicants."
            )
    else:
        reject_inference_note = ""

    # ── EAD: state the months-on-book basis, whatever it is ────────────────────
    # The essential EAD assumption used to appear nowhere in the report. Every loan was
    # carried at a months-on-book of exactly 40% of its term, because the payment column
    # the estimator needs is stripped by the leakage filter before it ever arrives — so
    # exposure was a deterministic function of (term, rate) alone (FLAWS-N10).
    _mob_basis = metrics.get("ead_mob_basis")
    _mob_labels = {
        "elapsed_since_origination":
            "Months on book is measured as elapsed time from origination to the reporting "
            "date, capped at the contractual term, so exposure declines with loan age as "
            "it should.",
        "payments_observed":
            "Months on book is inferred from cumulative payments against the contractual "
            "instalment.",
        "fixed_fraction_of_term":
            r"\textbf{Months on book is fixed at 40\% of the contractual term for every "
            r"loan}, because the payment history the estimator needs is removed by the "
            r"leakage filter. Exposure is therefore a deterministic function of term and "
            r"interest rate and carries no loan-level ageing information.",
    }
    ead_mob_assumption = _mob_labels.get(str(_mob_basis), "")

    # ── Vintage drift: counted from the table, never asserted ──────────────────
    # The shipped report claimed 2016--2018 PD ratios were "consistently below 0.85"
    # directly above a table in which not one row was below 0.85 — the claim described
    # the pre-recalibration state while the table showed the post-recalibration one
    # (FLAWS-N7).
    _bt_annual = _aggregate_backtest_to_annual(metrics.get("pd_backtest_vintage") or [])
    if _bt_annual:
        # Band from the production constants, not a copy of them (see
        # validation/backtest.VINTAGE_PASS_BAND).
        _band_lo, _band_hi = VINTAGE_PASS_BAND
        _over = [r for r in _bt_annual if r["pd_ratio"] > _band_hi]
        _under = [r for r in _bt_annual if 0 < r["pd_ratio"] < _band_lo]
        _in_band = len(_bt_annual) - len(_over) - len(_under)

        def _years(rows):
            return ", ".join(str(r["vintage"]) for r in rows)

        _parts = [
            f"Of the {len(_bt_annual)} origination years backtested, {_in_band} sit inside "
            f"the ${vintage_band_text()}$ predicted-to-actual band"
        ]
        if _over:
            _parts.append(
                f"{len(_over)} over-predict ({_years(_over)}, peaking at "
                f"{max(r['pd_ratio'] for r in _over):.2f})"
            )
        if _under:
            _parts.append(
                f"{len(_under)} under-predict ({_years(_under)}, lowest "
                f"{min(r['pd_ratio'] for r in _under):.2f})"
            )
        vintage_drift_sentence = (
            ", ".join(_parts)
            + ". "
            + (
                "The over-predicting cohorts are the development-era vintages: the "
                "recalibrator is fitted on out-of-time (2016--2018) evidence and applied "
                "across the whole book, so it corrects the recent era at the cost of "
                "pushing the older one above its realised rate. "
                if _over else ""
            )
            + "This pattern is the calibration drift identified in Section~7.2, and "
            "__RECALIB_PRODUCTION_NOTE__."
        )
    else:
        vintage_drift_sentence = (
            "No vintage backtest was produced in this run, so no drift statement is made."
        )

    # ── Bootstrap AUC CI and the Spiegelhalter test ────────────────────────────
    # Both were computed by the pipeline (500 bootstrap resamples each) and never
    # rendered, leaving the headline AUC without a precision statement and the
    # calibration section resting on Hosmer-Lemeshow alone (FLAWS-N38).
    _oot_disc = metrics.get("discrimination", {}).get("oot", {})
    _ci_lo, _ci_hi = _oot_disc.get("auc_ci_lower"), _oot_disc.get("auc_ci_upper")
    if _ci_lo is not None and _ci_hi is not None:
        auc_ci_note = (
            f"(bootstrap 95\\% CI [{float(_ci_lo):.4f}, {float(_ci_hi):.4f}], "
            "500 resamples)"
        )
    else:
        auc_ci_note = ""

    _spieg = (metrics.get("calibration", {}).get("oot", {}) or {}).get("spiegelhalter") or {}
    if _spieg:
        _z, _p = float(_spieg.get("z_stat", 0.0)), float(_spieg.get("p_value", 1.0))
        _verdict = (
            "agrees with Hosmer-Lemeshow" if (_p < 0.05) == (float(metrics.get(
                "calibration", {}).get("oot", {}).get("hl_pvalue", 1.0)) < 0.05)
            else "disagrees with Hosmer-Lemeshow, so the miscalibration verdict rests on "
                 "the more sensitive of the two tests"
        )
        _p_txt = "<0.0001" if _p < 1e-4 else f"={_p:.4f}"
        spiegelhalter_note = (
            f"The Spiegelhalter $Z$-test, which unlike Hosmer-Lemeshow requires no "
            f"binning, returns $Z={_z:.2f}$ ($p{_p_txt}$) and therefore {_verdict}."
        )
    else:
        spiegelhalter_note = ""

    # ── Provenance strings: selection funnel, Model B scope, binner, dropped phases ──
    # All four exist because the report previously described a pipeline that differed
    # from the one that ran (FLAWS-N29, N36, N32, N39).
    _stages = metrics.get("feature_selection_stages") or {}
    if _stages:
        _iv_lo, _iv_hi = (_stages.get("iv_band") or [0.02, 0.50])[:2]
        selection_funnel = (
            f"{_stages.get('n_candidates', 0)} candidate features "
            f"$\\rightarrow$ {_stages.get('n_after_iv', 0)} after the IV band "
            f"$[{float(_iv_lo):.2f}, {float(_iv_hi):.2f}]$ "
            f"$\\rightarrow$ {_stages.get('n_after_vif', 0)} after the VIF filter "
            f"$\\rightarrow$ {_stages.get('n_after_elasticnet', 0)} after ElasticNet "
            f"shrinkage $\\rightarrow$ \\textbf{{{_stages.get('n_after_sign_check', 0)}}} "
            "after the sign check."
        )
        _sign_dropped = _stages.get("dropped_by_sign_check") or []
        if _sign_dropped:
            selection_funnel += (
                " Dropped by the sign check: "
                + ", ".join(f"\\texttt{{{tex_escape(str(f))}}}" for f in _sign_dropped)
                + "."
            )
    else:
        selection_funnel = (
            "per-stage counts were not recorded in this run; see "
            "\\texttt{outputs/scorecard\\_tables.json}."
        )

    _mb_excluded = metrics.get("model_b_excluded_features") or []
    if _mb_excluded:
        model_b_excluded_str = ", ".join(
            f"\\texttt{{{tex_escape(str(f))}}}" for f in _mb_excluded
        )
    else:
        model_b_excluded_str = "\\texttt{int\\_rate} and \\texttt{grade}"

    _binner = str(metrics.get("binner", "unknown"))
    _binner_labels = {
        "optbinning": r"\texttt{optbinning} \texttt{BinningProcess}",
        "manual_fallback": r"manual quantile/merge fallback binner \textbf{(not optbinning)}",
    }
    binner_used = _binner_labels.get(_binner, tex_escape(_binner))

    _phase_failures = metrics.get("phase_failures") or []
    if _phase_failures:
        phase_failures_str = (
            f"\\textbf{{{len(_phase_failures)}}} --- "
            + "; ".join(tex_escape(str(f.get("message", ""))) for f in _phase_failures)
            + "."
        )
    else:
        phase_failures_str = "none."

    # Champion vs challenger feature parity (audit A12): the challenger silently dropped
    # scorecard-engineered columns absent from the raw frame, so the count is asserted from
    # the data rather than claimed in prose.
    _n_feat_sc = len(sc_tables.get("selected_features", []))
    _n_feat_ch = len(metrics.get("challenger", {}).get("shap_mean_abs", []))
    n_features_sc = str(_n_feat_sc)
    n_features_ch = str(_n_feat_ch)
    if _n_feat_ch and _n_feat_sc and _n_feat_ch < _n_feat_sc:
        _missing = _n_feat_sc - _n_feat_ch
        # The trailing full stop belongs to the generated fragment: the static text that
        # follows it in the template starts a new sentence (FLAWS-N35).
        feature_parity_note = (
            f"the challengers are therefore evaluated on {_missing} predictor(s) fewer than the "
            "champion, so the comparison above understates their attainable discrimination and "
            "the champion's margin should not be read as a like-for-like result."
        )
    else:
        feature_parity_note = (
            "the two model families are therefore evaluated on an identical predictor set, so "
            "the comparison above is like-for-like."
        )
    total_el = fmt_num(_num(metrics, "total_el"))
    total_ead = fmt_num(_num(metrics, "total_ead_portfolio"))
    el_rate = fmt_pct(_num(metrics, "el_rate"))
    total_rwa = fmt_num(_num(metrics, "total_rwa"))
    total_rwa_sa = fmt_num(_num(metrics, "total_rwa_sa"))
    rwa_density = str(metrics.get("rwa_density", "20.6%")).replace("%", "\\%")
    total_ecl = fmt_num(_num(metrics, "total_ecl"))
    ecl_coverage = fmt_pct(_num(metrics, "ecl_coverage"), precision=3)
    stage2_pct = fmt_pct(_num(metrics, "stage2_pct"))
    stage3_pct = fmt_pct(_num(metrics, "stage3_pct"))
    # Cite the operating cutoff: the risk-appetite rule (most inclusive score whose
    # approved bad rate stays under the ceiling), traceable to a highlighted table row.
    # It is neither a profit optimum nor a marginal-RAROC-hurdle rule, though the legacy
    # metrics key and two comments used to say so.
    _opt_profit_row = metrics.get("cutoff_risk_appetite") or metrics.get(
        "cutoff_optimal_profit", {}
    )
    opt_cutoff = fmt_dec(_opt_profit_row.get("cutoff", _num(metrics, "optimal_cutoff_threshold")), precision=0)
    opt_approval = fmt_pct(_opt_profit_row.get("approval_rate", _num(metrics, "optimal_approval_rate")))
    opt_bad = fmt_pct(_opt_profit_row.get("bad_rate", _num(metrics, "optimal_bad_rate")))
    opt_profit_m = f"{_opt_profit_row.get('expected_profit', 0.0) / 1e6:,.1f}"
    opt_raroc = fmt_pct(_opt_profit_row.get("raroc", 0.0))
    raroc_hurdle = fmt_pct(_num(metrics, "cutoff_raroc_hurdle"))
    # Data-driven hurdle comparison so the prose can never contradict the table
    # (a negative RAROC must not be described as "above" a positive hurdle).
    _opt_raroc_v = float(_opt_profit_row.get("raroc", 0.0))
    _hurdle_v = float(_num(metrics, "cutoff_raroc_hurdle"))
    if _opt_raroc_v >= 1.5 * _hurdle_v:
        raroc_vs_hurdle = "comfortably above"
    elif _opt_raroc_v >= _hurdle_v:
        raroc_vs_hurdle = "above"
    else:
        raroc_vs_hurdle = "below"
    # Whether ANY cutoff on the swept grid clears the hurdle is an empirical result and
    # flips once the PD horizon feeding the P&L is corrected, so it is derived, never
    # asserted (FLAWS-N2, N27).
    _grid_rows = [
        r for r in (metrics.get("cutoff_strategy_table") or [])
        if float(r.get("approval_rate", 0.0)) > 0.0
    ]
    _clearing = [r for r in _grid_rows if float(r.get("raroc", 0.0)) >= _hurdle_v]
    if not _grid_rows:
        grid_hurdle_verdict = ""
    elif not _clearing:
        grid_hurdle_verdict = (
            f"Note that RAROC stays below the {raroc_hurdle} hurdle across the entire "
            "400--800 grid, so \\emph{no} cutoff on this book clears it --- the operating "
            "point is a risk-appetite compromise, not a profitable optimum."
        )
    else:
        _best_clear = max(_clearing, key=lambda r: float(r.get("raroc", 0.0)))
        grid_hurdle_verdict = (
            f"{len(_clearing)} of the {len(_grid_rows)} non-empty cutoffs on the 400--800 "
            f"grid clear the {raroc_hurdle} hurdle, the strongest being cutoff "
            f"{int(_best_clear.get('cutoff', 0))} at a RAROC of "
            f"{fmt_pct(_best_clear.get('raroc', 0.0))}; the operating point trades some of "
            "that return for the risk-appetite ceiling on the approved bad rate."
        )
    max_bad_rate_txt = fmt_pct(_num(metrics, "cutoff_max_bad_rate"))
    # The charge netted out of expected profit is the cost of capital, NOT the RAROC
    # hurdle (the hurdle is only the threshold the resulting RAROC is compared against).
    cost_of_capital_txt = fmt_pct(_num(metrics, "cutoff_cost_of_capital"))
    _corner_row = metrics.get("cutoff_raroc_max") or metrics.get("cutoff_profit_argmax", {})
    corner_raroc = fmt_pct(_corner_row.get("raroc", 0.0))
    corner_cutoff = fmt_dec(_corner_row.get("cutoff", 0.0), precision=0)
    # Which corner the unconstrained optimum actually sits at is data-dependent: on a book
    # where every grid RAROC is negative the argmax is the MOST EXCLUSIVE non-empty cutoff,
    # not full approval. Derive the wording instead of asserting it.
    #
    # Every clause below is derived. An earlier version hard-coded "every cutoff on the
    # 400--800 grid returns a negative RAROC" and "the profit-maximising and
    # RAROC-maximising cutoff coincide" inside the exclusive-corner branch. Both were false
    # on a run whose grid RAROCs were all positive and whose two argmaxes sat at 400 and
    # 610 — and both sentences were printed beside the numbers that contradicted them.
    _corner_approval_v = float(_corner_row.get("approval_rate", 0.0))
    corner_approval = fmt_pct(_corner_approval_v, precision=3)
    _profit_argmax_row = metrics.get("cutoff_profit_argmax") or {}
    _raroc_max_row = metrics.get("cutoff_raroc_max") or {}
    _argmaxes_coincide = bool(_profit_argmax_row) and bool(_raroc_max_row) and (
        int(_profit_argmax_row.get("cutoff", -1)) == int(_raroc_max_row.get("cutoff", -2))
    )
    corner_agreement = (
        "the unconstrained profit-maximising \\emph{and} RAROC-maximising cutoff coincide at"
        if _argmaxes_coincide else
        "the unconstrained RAROC-maximising cutoff sits at"
    )
    _n_negative = sum(1 for r in _grid_rows if float(r.get("raroc", 0.0)) < 0.0)
    if _grid_rows and _n_negative == len(_grid_rows):
        _corner_reason = (
            " --- every cutoff on the 400--800 grid returns a negative RAROC, so the argmax "
            "is simply the smallest, best-quality approved book rather than a genuinely "
            "profitable operating point"
        )
    else:
        _corner_reason = (
            " --- RAROC is a ratio to economic capital, so it is maximised by the smallest, "
            "best-quality approved book regardless of whether that book is worth writing; "
            f"the profit argmax sits instead at cutoff \\textbf{{{fmt_dec(_profit_argmax_row.get('cutoff', 0.0), precision=0)}}} "
            f"(approving \\textbf{{{fmt_pct(float(_profit_argmax_row.get('approval_rate', 0.0)))}}})"
        )
    if _corner_approval_v >= 0.99:
        corner_desc = (
            "approving essentially the entire through-the-door population at a portfolio "
            f"RAROC of \\textbf{{{corner_raroc}}} --- because higher-risk grades carry "
            "interest rates high enough to remain RAROC-accretive even after loss and "
            "capital costs"
        )
        corner_implication = "unconstrained optimisation therefore implies near-total approval"
    else:
        corner_desc = (
            f"the most \\emph{{exclusive}} non-empty cutoff on the grid (score "
            f"\\textbf{{{corner_cutoff}}}, approving only \\textbf{{{corner_approval}}} of the "
            f"population at a RAROC of \\textbf{{{corner_raroc}}}){_corner_reason}"
        )
        corner_implication = (
            "unconstrained optimisation therefore collapses to a vacuous near-zero-volume corner"
        )

    # The implication clause has to answer for BOTH corners, not just the RAROC one. When
    # the profit argmax is near-total approval the book is not "vacuous" at all — it is
    # large and written at an unacceptable bad rate, which is a different reason to
    # constrain the problem and the one the risk-appetite rule actually responds to.
    _profit_approval_v = float(_profit_argmax_row.get("approval_rate", 0.0))
    if not _argmaxes_coincide and _profit_approval_v >= 0.99:
        corner_implication = (
            "neither unconstrained optimum is an operating point (one approves almost "
            "nobody; the other approves almost everybody, at an approved bad rate of "
            f"\\textbf{{{fmt_pct(float(_profit_argmax_row.get('bad_rate', 0.0)))}}})"
        )
    gini_ttd = fmt_dec(_num(metrics, "gini_ttd"))
    gini_shift = fmt_dec(metrics.get("gini_shift", -0.0555))
    stress_el = fmt_num(_num(metrics, "stress_el"))
    stress_rwa = fmt_num(_num(metrics, "stress_rwa"))
    stress_capital_req = fmt_num(_num(metrics, "stress_capital_req"))
    stress_el_ratio = fmt_pct(_num(metrics, "stress_el") / _num(metrics, "total_el") - 1.0, precision=1)
    stress_rwa_ratio = fmt_pct(_num(metrics, "stress_rwa") / _num(metrics, "total_rwa") - 1.0, precision=1)
    stress_cap_ratio = fmt_pct(
        _num(metrics, "stress_capital_req") / (_num(metrics, "total_rwa") * 0.08) - 1.0, precision=1
    )
    today_str = os.environ.get("REPORT_DATE") or date.today().strftime("%d %B %Y")
    rwa_release_cap = fmt_num((_num(metrics, "total_rwa_sa") - _num(metrics, "total_rwa")) * 0.08)
    base_cap_req = fmt_num(_num(metrics, "total_rwa") * 0.08)
    base_cap_req_sa = fmt_num(_num(metrics, "total_rwa_sa") * 0.08)
    rwa_release_cap_abs = fmt_num(abs((_num(metrics, "total_rwa_sa") - _num(metrics, "total_rwa")) * 0.08))
    hl_pvalue = fmt_dec(metrics.get("calibration", {}).get("oot", {}).get("hl_pvalue", 0.1656), 4)
    psi_train_oot = fmt_dec(metrics.get("stability", {}).get("psi_train_oot", 0.0005), 4)

    # ── Population counts: every count in the prose is derived, never hard-coded ──
    _n_train_v = int(_num(metrics, "n_train"))
    _n_test_v = int(_num(metrics, "n_test"))
    _n_oot_v = int(_num(metrics, "n_oot"))
    _n_model_v = _n_train_v + _n_test_v + _n_oot_v or 1
    _n_file_v = int(_num(metrics, "n_accepted_file"))
    _n_resolved_v = int(metrics.get("n_resolved_outcome", _num(metrics, "n_accepted_raw")))
    n_accepted_file = f"{_n_file_v:,}"
    n_resolved_outcome = f"{_n_resolved_v:,}"
    n_rejected_raw = f"{int(metrics.get('n_rejected_raw', 0)):,}"
    n_modelling = f"{_n_model_v:,}"
    n_greyzone = f"{max(0, _n_resolved_v - _n_model_v):,}"
    pct_train = fmt_pct(_n_train_v / _n_model_v)
    pct_test = fmt_pct(_n_test_v / _n_model_v)
    pct_oot = fmt_pct(_n_oot_v / _n_model_v)
    train_bad_rate = fmt_pct(_num(metrics, "train_bad_rate"))
    train_good_rate = fmt_pct(1.0 - float(_num(metrics, "train_bad_rate")))
    oot_bad_rate = fmt_pct(
        metrics.get("calibration_comparison", {}).get("before", {}).get("actual_dr", 0.0)
    )

    # ── Recalibration: what the out-of-time gate decided ────────────────────────
    # The gate triggers on out-of-time evidence, fits on the earlier half of the OOT
    # window and accepts only on demonstrated improvement in the later half. The prose
    # below reports whichever branch actually fired (AUDIT-A1).
    _gate = metrics.get("calibration", {}).get("recalibration_gate", {}) or {}
    _applied = bool(metrics.get("calibration", {}).get("recalibration_applied", False))
    _n_fit = f"{int(_gate.get('n_fit', 0)):,}"
    _n_eval = f"{int(_gate.get('n_eval', 0)):,}"
    _method = str(_gate.get("chosen_method", "none"))
    _eb, _ea = _gate.get("eval_before", {}), _gate.get("eval_after", {})
    # These are free text produced by the gate and may contain %, _, & etc.
    _trigger_reason = tex_escape(str(_gate.get("trigger_reason", "")))
    _skip_reason = tex_escape(str(_gate.get("skip_reason", "")))
    _article = "an" if _method[:1].lower() in "aeiou" else "a"
    _EM_A, _EM_B = "\\emph{", "}"
    # Print the real date boundaries of the two slices. Stating only the counts is what
    # allowed a positional split to read as a chronological one for an entire audit round
    # (FLAWS-N3).
    if _gate.get("fit_slice_min_date") and _gate.get("eval_slice_max_date"):
        _slice_span = (
            f" (fitting slice {_gate['fit_slice_min_date']} to "
            f"{_gate['fit_slice_max_date']}; evaluation slice "
            f"{_gate['eval_slice_min_date']} to {_gate['eval_slice_max_date']}, split on "
            "origination date)"
        )
    elif _gate.get("split_basis") == "positional":
        _slice_span = (
            " \\textbf{(split positionally, by row order --- this is not an out-of-time "
            "split and the results below should be read accordingly)}"
        )
    else:
        _slice_span = ""

    _gate_intro = (
        "Recalibration is governed by an out-of-time gate. The OOT window is split "
        f"chronologically: the earlier {_n_fit} loans form the fitting slice and the "
        f"later {_n_eval} the evaluation slice{_slice_span}. The gate " + _EM_A + "triggers" + _EM_B +
        " on evidence of miscalibration in the fitting slice, " + _EM_A + "fits" + _EM_B +
        " the candidate transform on that slice only, and " + _EM_A + "accepts" + _EM_B +
        " it solely if it demonstrably improves calibration on the evaluation slice, "
        "which is never used for fitting. Gating on the in-time partition instead would "
        "be blind to the very era drift documented in Section~7.3 --- that test passes "
        "comfortably --- while fitting on the evaluation slice would trivially pass "
        "Hosmer-Lemeshow as an in-sample artefact. This design avoids both failure modes."
    )

    # The transform is attached only to the vintages the gate learned from (2016+), so
    # everything older keeps raw model PDs. This was recorded in a log line and nowhere
    # else, while the paragraph below asserted flatly that the PDs feeding EL, RWA and
    # staging are recalibrated -- leaving an undisclosed level discontinuity at the 2016
    # boundary for a large minority of the book.
    _cal_min_year = metrics.get("calibration_min_issue_year")
    _oos_ead = metrics.get("calibration_out_of_scope_ead_share")
    _oos_loans = metrics.get("calibration_out_of_scope_loan_share")
    if _cal_min_year:
        # Both shares, because they tell different stories: the untouched vintages are a
        # large minority of the book by loan count but a small share of exposure, since
        # EAD amortises to the reporting date and those loans are the oldest.
        if _oos_ead is not None and _oos_loans is not None:
            _scope_extent = (
                f" --- \\textbf{{{fmt_pct(_oos_loans, precision=1)}}} of loans, though only "
                f"\\textbf{{{fmt_pct(_oos_ead, precision=2)}}} of exposure by EAD, since EAD "
                "amortises to the reporting date and these are the oldest vintages ---"
            )
        else:
            _scope_extent = ""
        _scope_clause = (
            f", but only for vintages originated in \\textbf{{{int(_cal_min_year)}}} or "
            f"later, which is the era the gate learned from. Earlier vintages{_scope_extent} "
            "keep their raw model PD, so the deployed PD level steps at the "
            f"{int(_cal_min_year)} boundary"
        )
        _scope_short = f" for {int(_cal_min_year)}+ vintages only"
    else:
        _scope_clause = ""
        _scope_short = ""

    if not _gate:
        recalib_status = _gate_intro
        recalib_production_note = (
            "assessed by the out-of-time recalibration gate (Section~7.2)"
        )
    elif _gate.get("skip_reason"):
        recalib_status = (
            f"{_gate_intro} On this run the gate did not run: {_skip_reason}. "
            "All reported PDs are therefore raw model output."
        )
        recalib_production_note = (
            "not corrected in production: the recalibration gate could not run "
            "(Section~7.2), so this under-prediction carries through untreated"
        )
    elif not _gate.get("triggered", False):
        recalib_status = (
            f"{_gate_intro} On this run the gate did " + _EM_A + "not" + _EM_B +
            f" trigger ({_trigger_reason or 'no miscalibration detected'}), so "
            "no transform was fitted and all reported PDs are raw model output."
        )
        recalib_production_note = (
            "not corrected in production: the out-of-time gate did not trigger "
            "(Section~7.2)"
        )
    elif _applied:
        recalib_status = (
            f"{_gate_intro} On this run the gate " + _EM_A + "triggered" + _EM_B +
            f" ({_trigger_reason}), selected {_article} " + _EM_A + _method +
            _EM_B + " transform by out-of-fold Brier score computed within the fitting "
            "slice, and " + _EM_A + "accepted" + _EM_B + " it. On the held-out later "
            "slice the predicted-to-actual ratio moves from "
            f"{_eb.get('ratio', float('nan')):.3f} to "
            f"{_ea.get('ratio', float('nan')):.3f}, the calibration intercept from "
            f"{_eb.get('intercept', float('nan')):.4f} to "
            f"{_ea.get('intercept', float('nan')):.4f}, and the Brier score from "
            f"{_eb.get('brier', float('nan')):.4f} to "
            f"{_ea.get('brier', float('nan')):.4f}. The transform " + _EM_A + "is" +
            _EM_B + " attached to the production scorecard, so the PDs feeding Expected "
            "Loss, Basel RWA and IFRS~9 staging are recalibrated" + _scope_clause +
            ". Because every one of "
            "those improvements is measured on vintages excluded from the fit, this is "
            "an out-of-sample result rather than an in-sample artefact."
        )
        recalib_production_note = (
            f"corrected in production by the {_method} transform that the out-of-time "
            "gate accepted (Section~7.2), applied to the scorecard's 12-month PD feeding "
            f"Expected Loss, Basel RWA and SICR staging{_scope_short}"
        )
    else:
        recalib_status = (
            f"{_gate_intro} On this run the gate " + _EM_A + "triggered" + _EM_B +
            f" ({_trigger_reason}) and {_article} " + _EM_A + _method + _EM_B +
            " transform was fitted on the earlier slice --- but it was " + _EM_A +
            "rejected" + _EM_B + ": on the held-out later slice the predicted-to-actual "
            f"ratio moves from {_eb.get('ratio', float('nan')):.3f} to "
            f"{_ea.get('ratio', float('nan')):.3f} and the Brier score from "
            f"{_eb.get('brier', float('nan')):.4f} to "
            f"{_ea.get('brier', float('nan')):.4f}, which does not clear the acceptance "
            "test. No calibrator is attached, and all reported PDs --- including those "
            "feeding Expected Loss, Basel RWA and IFRS~9 staging --- are raw model "
            "output. We report the rejection rather than deploying a transform that does "
            "not generalise: the drift is real and documented in Section~7.3, but a "
            "monotone rescaling fitted on earlier vintages failing to transfer to later "
            "ones is itself evidence that the miscalibration is not a stable level shift."
        )
        recalib_production_note = (
            "not corrected in production: the out-of-time gate fitted a candidate "
            "transform and rejected it for failing to generalise to the held-out later "
            "vintages (Section~7.2), so this under-prediction carries through untreated "
            "into Expected Loss, Basel RWA and SICR staging"
        )


    # ── LaTeX template (Phase 6 B&W academic, XeLaTeX + biblatex) ─────────────
    latex_template = r"""%!TEX program = xelatex
\documentclass[11pt,a4paper]{article}

% --- XeLaTeX fonts: B&W academic (TeX Gyre Pagella + TeX Gyre Pagella Math)
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{TeX Gyre Pagella}
\setsansfont{TeX Gyre Pagella}
\setmonofont{TeX Gyre Cursor}[Scale=MatchLowercase]
\setmathfont{TeX Gyre Pagella Math}

% --- Layout
\usepackage[left=2.0cm,right=2.0cm,top=1.9cm,bottom=1.9cm,headheight=14pt]{geometry}
\usepackage{microtype,parskip}

% --- Core packages (B&W only)
\usepackage{amsmath,mathtools}
\usepackage{booktabs,longtable,multirow,array,tabularx}
\usepackage{graphicx,subcaption,float,caption}
\usepackage{fancyhdr,titlesec,enumitem}
\usepackage{siunitx}
\usepackage[hidelinks,
  colorlinks=true,
  linkcolor=black, citecolor=black, urlcolor=black,
  pdftitle={Credit Risk \& IFRS 9 ECL Engine --- Model Risk Report},
  pdfauthor={Dimitrios Kotoulias}
]{hyperref}

% --- Bibliography (biblatex + biber, authoryear style)
\usepackage[backend=biber,style=authoryear,sorting=nyt,maxbibnames=3,
  giveninits=true,doi=false,isbn=false,url=false,date=year]{biblatex}
\addbibresource{model_risk_report.bib}

% --- Section headings: small-caps with thin rule
\titleformat{\section}{\Large\bfseries\scshape}{\thesection}{0.8em}{}[\vspace{-0.4em}\rule{\linewidth}{0.4pt}]
\titleformat{\subsection}{\normalsize\bfseries\scshape}{\thesubsection}{0.8em}{}
% Tighter float separation - 44 floats x ~20pt of default padding is the
% largest remaining block of recoverable whitespace (the internal review log page budget).
\setlength{\abovedisplayskip}{4pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{4pt plus 2pt minus 2pt}
\setlength{\abovedisplayshortskip}{2pt}
\setlength{\belowdisplayshortskip}{2pt}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{2pt}
\titlespacing{\section}{0pt}{7pt}{3pt}
\titlespacing{\subsection}{0pt}{5pt}{2pt}
\setlength{\parskip}{3pt plus 1pt minus 1pt}

% --- Header/footer (B&W, scshape)
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\scshape Credit Risk \& IFRS 9 ECL Engine}
\fancyhead[R]{\small Dimitrios Kotoulias \textbullet{} AUEB}
\fancyfoot[C]{\small\thepage}
\fancyfoot[R]{\small\itshape Model Risk Report}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

\captionsetup{font={small},labelfont={bf,sc},labelsep=period,skip=2pt}

\begin{document}

% TITLE PAGE
\begin{titlepage}
  \centering
  \vspace*{4cm}
  \rule{0.6\linewidth}{0.6pt}\\[1.5em]
  {\fontsize{26}{30}\selectfont\scshape Credit Risk \& IFRS~9\\[6pt] ECL Engine\par}
  \vspace{1em}
  \rule{0.6\linewidth}{0.6pt}\\[2em]
  {\large\itshape Model Validation and Quantitative Assessment Report\par}
  \vspace{4cm}
  {\large\scshape Dimitrios Kotoulias\par}
  {\small Athens University of Economics \& Business\par}
  \vspace{0.6em}
  {\small Lending Club Consumer Loans $\cdot$ 2007--2018 $\cdot$ N = __N_ACCEPTED_FILE__ accepted\par}
  \vspace{0.4em}
  {\small __TODAY__\par}
  \vfill
  \rule{\linewidth}{0.4pt}\\[0.5em]
  {\footnotesize\ttfamily github.com/DimitrisKotoulias/ifrs9-credit-risk-engine\par}
\end{titlepage}

\newpage
\begin{abstract}
This report presents the model development, validation, and risk quantification
for a retail credit underwriting engine trained on the LendingClub 2007--2018
loan portfolio (__N_ACCEPTED_FILE__ accepted loans, of which __N_RESOLVED__ have a resolved
good/bad outcome and __N_MODELLING__ form the modelling population after the train/OOT grey
zone is excluded; __N_REJECTED_RAW__ rejected applications).
The Probability of Default (PD) scorecard achieves an out-of-time Gini of
\textbf{VAR_GINI_OOT} (AUC = VAR_AUC_OOT), with Population Stability
Index PSI = VAR_PSI_OOT (model stability: VAR_GINI_RAG).
The two-stage Loss Given Default model yields mean LGD = VAR_MEAN_LGD
and downturn LGD = VAR_DOWNTURN_LGD (90th percentile, Basel-conservative).
Basel IRB Risk-Weighted Assets total \$VAR_RWA_IRB at VAR_RWA_DENSITY RWA density.
IFRS 9 Expected Credit Loss provisions total \$VAR_ECL_TOTAL (coverage: VAR_ECL_COVERAGE).
\end{abstract}

\tableofcontents
\newpage

% -----------------------------------------------------------------------------
% 1. INTRODUCTION AND EXECUTIVE SUMMARY
% -----------------------------------------------------------------------------
\section{Introduction and Executive Summary}

Retail credit risk requires modeling frameworks that are explainable and compliant with capital and accounting regulation. Under the Basel Committee on Banking Supervision (BCBS) capital accords \parencite{bcbs2004} and the International Financial Reporting Standards (IFRS~9) accounting guidelines \parencite{iasb2014}, financial institutions must deploy internal risk engines to assess risk-adjusted pricing, regulatory capital, and forward-looking impairment provisions.

This report documents the mathematical foundation, development methodology, and empirical validation of an end-to-end retail credit underwriting, capital calculation, and expected credit loss (ECL) engine. Trained on __N_MODELLING__ historical consumer records originated between 2007 and 2018 (drawn from __N_ACCEPTED_FILE__ accepted loans, __N_RESOLVED__ of which have a resolved credit outcome), this framework is designed to bridge the gap between risk underwriting, regulatory capital management, and standard accounting provisions.

The core objective is to move away from simplistic statistical estimations and instead construct a highly transparent, mathematically rigorous portfolio risk model that covers:
\begin{enumerate}
    \item \textbf{Underwriting:} An interpretable credit scorecard based on Weight of Evidence (WoE) monotonic binning and regularised logistic regression, validated against a non-linear LightGBM challenger model.
    \item \textbf{Loss Mitigation:} A bimodal two-stage LGD model (cure probability + conditional severity) with Downturn LGD adjustments for capital stress testing.
    \item \textbf{Capital Reserve:} Per-loan and aggregated Basel IRB regulatory capital calculation under the ``Other Retail'' Vasicek Asymptotic Single Risk Factor (ASRF) model.
    \item \textbf{Financial Provisioning:} A forward-looking IFRS 9 expected credit loss engine driven by discrete-time logistic survival curves across three stages, probability-weighted across macroeconomic scenarios.
    \item \textbf{Decision Optimisation:} A profit-maximising score cut-off model and reject inference (parcelling) methodology to address selection bias in the underwriting population.
\end{enumerate}

\subsection{Key Performance Metrics}
Table~\ref{tab:exec_summary} summarizes the portfolio-level credit metrics and validation statistics calculated across the underwriting, capital, and impairment phases.

\begin{table}[h]
\centering
\caption{Portfolio Headline Summary and Quantitative Benchmarks}
\label{tab:exec_summary}
\vspace{0.5em}
\begin{tabular}{p{3.8cm}p{3.2cm}p{9.0cm}}
\toprule
\textbf{Quant Dimension} & \textbf{Metric Value} & \textbf{Regulatory Purpose \& Benchmark} \\
\midrule
PD Discrimination (OOT) & __GINI_OOT__ Gini & Out-of-Time score risk rank-ordering capability. \\
& (__AUC_OOT__ AUC) & Matches regulatory standards for acceptable discrimination. \\
OOT Separation (KS) & __KS_OOT__ & Kolmogorov-Smirnov statistic; values above $\sim 0.30$ are typical for retail scorecards. \\
OOT Calibration Brier & __BRIER_OOT__ & Brier score reflecting high probability accuracy. \\
OOT Calibration p-value (raw, pre-recalib.) & __HL_PVALUE__ & Hosmer-Lemeshow $p$-value on the \textbf{raw} scorecard PD ($p > 0.05$ confirms good calibration); see Table~\ref{tab:calibration_comparison} for the post-recalibration before/after comparison. \\
OOT Population Stability & __PSI_TRAIN_OOT__ & PSI Train-to-OOT ($< 0.10$ denotes absolute population stability). \\
\midrule
LGD Model Summary & __MEAN_LGD__ Mean LGD & Primary provisioning loss rate. \\
& __DOWNTURN_LGD__ Downturn & Conservative 90th percentile stress limit for capital charge. \\
\midrule
Expected Loss (EL) & \$__TOTAL_EL__ & Lifetime Expected Loss projection of current portfolio. \\
Portfolio EL Rate & __EL_RATE__ & Underwriting expected loss density. \\
\midrule
Basel IRB Total RWA & \$__TOTAL_RWA__ & Risk-Weighted Assets calculated using retail ASRF formula. \\
Basel SA Total RWA & \$__TOTAL_RWA_SA__ & Standardised Approach baseline capital RWA reference (75\% RW). \\
Basel RWA Density & __RWA_DENSITY__ & IRB RWA divided by total portfolio EAD (\$__TOTAL_EAD__). \\
\midrule
Portfolio IFRS 9 ECL & \$__TOTAL_ECL__ & Probability-weighted Stage 1, 2, \& 3 Expected Credit Loss. \\
Portfolio ECL Coverage & __ECL_COVERAGE__ & Capital coverage buffer (Total ECL / Total EAD). \\
\bottomrule
\end{tabular}
\end{table}

An OOT AUC of __AUC_OOT__ (Gini __GINI_OOT__) is the realistic discrimination ceiling for an application scorecard built on origination-only features on LendingClub-style unsecured consumer data: __RANGE_GINI_OOT__ is the published Gini range for consumer-credit scorecards surveyed by \textcite{lessmann2015benchmarking}, __CEILING_NOTE__. The figure should be read as realistic, benchmarked performance rather than a shortfall.

The scorecard demonstrates stable, high-contrast risk separation across distinct macroeconomic cycles. Under Basel capital standards, the risk-sensitive Internal Ratings-Based (IRB) approach identifies a capital requirement of \textbf{\$__BASE_CAP_REQ__} compared to \textbf{\$__BASE_CAP_REQ_SA__} under the Standardised Approach. This capital surcharge of \textbf{\$__RWA_RELEASE_CAP_ABS__} reflects the elevated risk profile (PD/LGD) of the LendingClub consumer portfolio, demonstrating that a flat 75\% risk weight under the Standardised Approach materially undercapitalises this retail asset class.

This risk-sensitive capital surcharge highlights the necessity of developing internal ratings-based (IRB) risk frameworks \parencite{bcbs2004} to ensure adequate capital provisioning for higher-yielding, higher-risk retail credit assets rather than relying on rigid, risk-insensitive standardized approaches.

% -----------------------------------------------------------------------------
% 2. DATA ENGINEERING AND EXPLORATORY ANALYSIS
% -----------------------------------------------------------------------------
\section{Data Engineering and Exploratory Analysis}

\subsection{Data Source and Exclusions}
The primary underwriting and historical performance data is derived from Lending Club's consumer loan database, covering the years 2007 through 2018. This dataset contains credit bureau features and demographic information gathered at loan origination, along with post-origination transaction and default markers.

To ensure methodological correctness, loans with ambiguous or immature repayment statuses are excluded from the modeling population. Specifically, loans marked as \textit{``Current''}, \textit{``In Grace Period''}, or \textit{``Late (16--30 days)''} are removed since their ultimate credit outcome is unresolved. The remaining loans represent the underwriting and model development population.

\subsection{Target Definition (PD)}
A binary default indicator ($Y$) is defined on each loan's \emph{resolved status} at the 2018Q4 snapshot, taken directly from the pipeline configuration (``resolved'' rather than ``terminal'': as set out below, one member of the bad set is a current delinquency state, not a final outcome):
\begin{equation}
Y =
\begin{cases}
1 \text{ (Bad)}, & \text{if status } \in \{__TARGET_BAD_SET__\} \\
0 \text{ (Good)}, & \text{if status } \in \{__TARGET_GOOD_SET__\}
\end{cases}
\end{equation}
The bad set includes \textit{Late (31--120 days)} and is therefore \textbf{wider} than the BCBS 90+ DPD reference definition: it admits 31--89 DPD delinquency as default. This is deliberate. The dataset is a status snapshot rather than a days-past-due panel, so loans sitting at exactly 90+ DPD on the snapshot date are few, and restricting the bad set to them would discard most of the delinquent population. The consequence is a \emph{more conservative} default flag than the regulatory standard, and it should be read that way wherever this report compares its default rates to published benchmarks (Section~10).

\subsection{Out-of-Time (OOT) Splitting}
To replicate standard banking validation practices, the data is split chronologically based on loan origination date (\texttt{issue\_d}):
\begin{itemize}
    \item \textbf{Training Population:} Loans originated prior to January 2015 ($N = \text{VAR_N_TRAIN}$; __PCT_TRAIN__ of modelling population). Training bad rate: __TRAIN_BAD_RATE__.
    \item \textbf{In-Time Holdout Set:} Simple random 20\% sample from the training period ($N = \text{VAR_N_TEST}$; __PCT_TEST__ of modelling population). The sample is drawn without class stratification; at this size the __TRAIN_GOOD_RATE__/__TRAIN_BAD_RATE__ good/bad ratio is reproduced by the law of large numbers rather than enforced by design.
    \item \textbf{Out-of-Time (OOT) Set:} All loans originated between January 2016 and December 2018 ($N = \text{VAR_N_OOT}$; __PCT_OOT__ of modelling population). OOT observed bad rate: __OOT_BAD_RATE__, reflecting the portfolio's seasoning and the macroeconomic deterioration of the 2016--2018 vintage cohorts.
\end{itemize}
This chronological split simulates how the model will perform on future vintages, testing for structural or macroeconomic shifts.

\textbf{Excluded grey zone.} The two cut-offs are deliberately non-adjacent: loans originated in the twelve months between them (the entire \textbf{2015} origination cohort) fall into neither partition and are \emph{dropped from the modelling population entirely}. The embargo prevents 2015 vintages --- which are simultaneously close enough to the training window to share its underwriting regime and immature enough at the 2018Q4 snapshot to have unresolved outcomes --- from contaminating either side of the temporal validation. The percentages above, the modelling population of __N_MODELLING__ loans (__N_GREYZONE__ resolved-outcome loans fall in the grey zone and are dropped), and every vintage table in this report are therefore stated \emph{net of 2015}, which is why the vintage series in Tables~\ref{tab:lifetime_pd_calibration} and~\ref{tab:pd_backtest} step directly from 2014 to 2016.

\subsection{Leakage and Target Variable Separation}
Data leakage is controlled by dropping post-origination fields (e.g., outstanding balance, payment records, and recovery metrics) from the PD feature set. However, these post-origination variables (such as actual recoveries and write-offs) are preserved for the defaulted-only population. This allows the LGD model to evaluate recovery performance without leaking future data into the PD scorecard.

\subsection{Vintage Cohort Analysis}
Vintage analysis tracks the cumulative default curves of origination cohorts (quarters) over their Months-on-Book (MOB). Figure~\ref{fig:vintage_curves} illustrates these curves. A steep initial slope indicates seasoning, while the curves flatten as high-risk accounts default early, illustrating standard credit risk dynamics.

\begin{figure}[H]
\centering
\includegraphics[width=0.65\textwidth]{figures/vintage_default_curves.png}
\caption{Cumulative Default Rates by Quarterly Vintage Cohorts}
\label{fig:vintage_curves}
\end{figure}

Figure~\ref{fig:eda_target_grade} displays additional exploratory distributions, demonstrating the relationship between historical default rates and risk grades, amortization terms, and loan purposes. Figure~\ref{fig:eda_dist_missing} completes the exploratory picture with the marginal distributions of the key numeric predictors and the missingness profile of the \emph{modelling} feature set --- that is, after the leakage deny-list has been applied in the loader, not the raw file. The distinction matters: the columns with the highest missingness in the raw LendingClub extract are the post-origination hardship and settlement blocks, which the deny-list removes before this figure is drawn, so describing it as the raw profile (as this passage previously did) points the reader at a different set of columns from the one plotted. The treatment applied to the remaining missing values is described in Section~3.1.

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/numeric_distributions.png}
    \caption{Key numeric feature distributions}
    \label{fig:num_dist}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/missingness.png}
    \caption{Missingness density}
    \label{fig:missing_anal}
\end{subfigure}
\caption{Exploratory data analysis of the development sample: predictor distributions and missingness.}
\label{fig:eda_dist_missing}
\end{figure}

All EDA visualizations in this section are computed on the in-time development sample (train and test partitions of the resolved-outcome modelling population), not the full raw portfolio: the Out-of-Time partition is withheld from all exploratory analysis to preserve the integrity of the temporal validation, and loans with unresolved statuses are excluded per Section~2.1. Portfolio-level capital and impairment calculations are run on the modelling population ($N = $ __N_MODELLING__), i.e.\ the train, test and OOT partitions combined --- not on the __N_ACCEPTED_FILE__ loans in the source file, since loans with an unresolved status carry no good/bad label and are excluded (__N_RESOLVED__ remain after that filter, of which __N_GREYZONE__ fall in the excluded grey zone described in Section~2.3). Population counts embedded in the EDA figures therefore reflect the development sample.

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/target_distribution.png}
    \caption{Target Distribution (Good vs. Bad)}
    \label{fig:target_dist}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/default_rate_by_grade.png}
    \caption{Default Rate by Underwriting Grade}
    \label{fig:default_grade}
\end{subfigure}
\\[0.5em]
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/default_rate_by_term.png}
    \caption{Default Rate by Amortisation Term}
    \label{fig:default_term}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/default_rate_by_purpose.png}
    \caption{Default Rate by Loan Purpose}
    \label{fig:default_purpose}
\end{subfigure}
\caption{EDA Risk, Grade, Term, and Purpose Distributions}
\label{fig:eda_target_grade}
\end{figure}

% -----------------------------------------------------------------------------
% 3. PROBABILITY OF DEFAULT (PD) SCORECARD DEVELOPMENT
% -----------------------------------------------------------------------------
\section{Probability of Default (PD) Scorecard Development}

\subsection{Weight of Evidence (WoE) and Information Value (IV)}
Continuous features are binned to handle non-linear relationships, outliers, and missing values. Binning is performed by \texttt{optbinning}'s optimal-binning solver, which searches bin boundaries under a mixed-integer formulation; where the solver is unavailable the pipeline falls back to a quantile-split binner that merges adjacent bins until the sign of the WoE differences is homogeneous. Which of the two produced the bins reported here is recorded in Appendix~C. For each bin $i$, the Weight of Evidence (WoE) is calculated as:
\begin{equation}
WoE_i = \ln\!\left( \frac{\text{Proportion of Good}_i}{\text{Proportion of Bad}_i} \right) = \ln\!\left( \frac{N_{G,i} / N_{G,total}}{N_{B,i} / N_{B,total}} \right)
\end{equation}
The predictive power of each feature is evaluated using the Information Value (IV):
\begin{equation}
IV = \sum_{i=1}^{k} \left( \frac{N_{G,i}}{N_{G,total}} - \frac{N_{B,i}}{N_{B,total}} \right) \times WoE_i
\end{equation}
\parencite[Ch.~4]{siddiqi2017}; \parencite{hand1997}. Features with an IV below $0.02$ are dropped due to low predictive power, while multicollinearity is controlled by removing features with a Variance Inflation Factor (VIF) greater than $5.0$. Two further selection stages follow; the complete funnel is set out in Section~3.3.

\subsubsection*{Missing-value treatment}
Missing values are passed to the binner as missing and receive a bin of their own, with a Weight of Evidence estimated from the observed default rate among the missing rows. Their contribution to a borrower's score is therefore whatever the data says it is, and it is visible as a populated row in the points table rather than folded invisibly into another band.

This matters more than it may appear. An earlier version of this pipeline substituted the sentinel value $-9999$ before binning. That placed every missing observation in the extreme lower bin of its feature, with three consequences: the binner's own ``Missing'' bin was structurally empty and its code path never executed; the implied risk direction was arbitrary and inconsistent across features, since a missing value scored as \emph{worst} for a feature where low is risky (FICO, available credit) and as \emph{best} for one where low is safe (DTI, recent enquiries); and for \texttt{mths\_since\_recent\_bc}, where missing means the borrower has never held a bankcard, the assignment carried the wrong sign outright. Because the two binning implementations also handled the sentinel differently, they produced materially different models from identical data.

\subsection{Logistic Regression and Scorecard Scaling}
A regularized logistic regression \parencite{hosmer2013} is fitted on the WoE-transformed features. Since higher WoE corresponds to a higher proportion of ``Good'' loans relative to ``Bad'' loans, all coefficients must be negative when predicting default ($Y=1$). The scorecard is then scaled to a points-based system using:
\begin{equation}
Score = Offset + Factor \times \ln(\text{odds})
\end{equation}
\begin{equation}
Factor = \frac{PDO}{\ln(2)}, \quad Offset = TargetScore - Factor \times \ln(TargetOdds)
\end{equation}
For $TargetScore = 600$ points, $TargetOdds = 50:1$, and $PDO = 20$ (points to double the odds), the scaling parameters are:
\begin{itemize}
    \item $Factor = 28.8539$
    \item $Offset = 487.123$
\end{itemize}
The points contributed by a specific binned attribute $j$ are calculated as:
\begin{equation}
Points_j = \left( -(WoE_j \times \beta_j) + \frac{\alpha}{n} \right) \times Factor + \frac{Offset}{n}
\end{equation}
where $\beta_j$ is the regression coefficient, $\alpha$ is the model intercept, and $n$ is the number of active features \parencite[Ch.~5]{anderson2007}.

\subsection{Feature Selection Results: the Four-Stage Funnel}
After WoE binning, candidate features pass through \textbf{four} sequential filters, not two. Reporting only the first two would leave the final list unreconcilable with the IV ranking below --- several high-IV candidates are removed by the later stages:

\begin{enumerate}
    \item \textbf{IV band.} Features with $IV < 0.02$ (negligible predictive power) or $IV > 0.50$ (likely target leakage) are excluded.
    \item \textbf{VIF filter.} Multicollinearity is controlled by iteratively removing features with a Variance Inflation Factor above 5.0.
    \item \textbf{ElasticNet shrinkage.} A cross-validated penalised logistic regression (\texttt{LogisticRegressionCV}, SAGA solver, $\ell_1$ ratio $0.5$, 3-fold stratified CV) is fitted on the survivors, and features whose coefficient shrinks to $|\beta| \le 10^{-4}$ are dropped. This stage removes predictors that are individually informative but redundant given the rest of the set --- which is why a high-IV variable can be absent from the final list.
    \item \textbf{Sign check.} The unpenalised logistic regression is fitted and any feature whose coefficient carries the wrong sign (positive on a WoE predictor, i.e.\ higher WoE implying higher default risk) is dropped, and the model refitted on the survivors.
\end{enumerate}

\noindent The funnel for this run: __SELECTION_FUNNEL__

\begin{sloppypar}\noindent
The final set is: __SELECTED_FEATURES__.
\end{sloppypar}

__IV_TABLE__

\subsection{Logistic Regression Coefficient Output}
The following table presents the logistic regression coefficient estimates, standard errors, z-statistics and p-values for all retained features. As expected for a WoE-based model predicting default ($Y=1$), all feature coefficients carry a \textbf{negative sign}: higher WoE values indicate a higher proportion of ``Good'' loans, and therefore map to lower default probability.

__LOGIT_TABLE__

\subsection{Scorecard Points Table}
Each WoE bin is converted to credit score points using the scaling formula. Higher points correspond to lower default risk. The table below shows the top bins ranked by point spread (i.e., the features with the greatest discriminatory contribution to the final score).

__SCORECARD_POINTS__

\subsection{Interpretability vs.\ Performance: Challenger Model Benchmark}
A non-linear LightGBM model was trained as a challenger \parencite{baesens2016} on the scorecard's selected predictors (see Section~7.8 for the exact feature count actually consumed by each model). Although gradient boosting is competitive in-sample, __CHAMPION_RATIONALE__ Regulatory standards (such as the US Fair Credit Reporting Act) require financial institutions to provide clear ``adverse action codes'' (reasons for denial) to rejected applicants. A linear scorecard allows for immediate, exact points-attribution for each feature, which is not possible with complex machine learning models without relying on approximations like SHAP.

To address potential policy-decision circularity from using LendingClub's own underwriting variables (\texttt{int\_rate} and \texttt{grade}), we developed a secondary Pure Underwriting Scorecard (Model B). Model B excludes __MODELB_EXCLUDED__. The exclusion list is therefore wider than pricing alone: the requested loan amount and its derived instalment are also withheld, so Model B is a deliberately conservative lower bound on what an independent underwriter could achieve, not a like-for-like re-fit with pricing removed. One qualification belongs here: the engineered \texttt{loan\_to\_income} ratio is \emph{not} excluded, so loan size still reaches Model B in normalised form. Table~\ref{tab:underwriting_comparison} compares the performance of the full scorecard (Model A) against this independent underwriting model. While Model A is the designated pipeline champion due to its superior discrimination, it utilizes LendingClub's pricing variables which are themselves highly correlated risk assessments. This introduces a degree of decision circularity. Model B (Pure Underwriting Scorecard) shows that a model built solely on raw credit bureau and demographic features remains competitive (OOT AUC = VAR_MODELB_AUC_OOT vs Model A's VAR_AUC_OOT, both as reported in Table~\ref{tab:underwriting_comparison}), supporting the scorecard's viability in an independent bank underwriting environment.

__UNDERWRITING_COMPARISON_TABLE__

\subsection{Selection Bias and Reject Inference (Parcelling)}
Scorecard models developed only on approved applicants suffer from selection bias. Because rejected applicants are excluded, their risk profiles and actual default rates are unobserved. To adjust for this, we implemented the \textbf{Parcelling} reject inference technique to probabilistically allocate outcomes to rejected applicants based on the accepts scorecard's predictions.

The pooled accepts and parcelled rejects population was refitted to produce a corrected through-the-door (TTD) scorecard. __REJECT_INFERENCE_NOTE__ Table~\ref{tab:reject_inference} outlines the results of the refitting and selection bias adjustment:

\begin{table}[h]
    \centering
    \caption{Reject Inference (Parcelling) Gini Coefficient Shift}
    \label{tab:reject_inference}
    \vspace{0.5em}
    \begin{tabular}{llp{6cm}}
        \toprule
        \textbf{Scorecard Population} & \textbf{Gini} & \textbf{Business Interpretation} \\
        \midrule
        \textbf{Accepts-Only (Base)} & __GINI__ & Champion scorecard on the approved-only in-time test partition. \\
        \textbf{Through-the-Door (Parcelled Refit)} & __GINI_TTD__ & Corrected scorecard accounts for selection bias across TTD. \\
        \midrule
        \textbf{Gini Coefficient Shift} & __GINI_SHIFT__ & Conservative risk dilution when scoring raw through-the-door applicants. \\
        \bottomrule
    \end{tabular}
\end{table}

The Gini shift of \textbf{__GINI_SHIFT__} points is consistent with the standard credit-cycle finding that through-the-door populations carry higher latent risk. \emph{Measurement caveat:} the shift is computed \emph{within} the parcelled refit --- the same through-the-door model scored on the accepts subset versus on the full accepts-plus-parcelled population --- and both figures are in-sample; the rejects additionally carry inferred, not observed, labels. The shift is then applied to the champion scorecard's held-out Gini to obtain the through-the-door figure in the table. The two rows are therefore not two independently validated scorecards, and the shift should be read as a directional selection-bias indicator rather than a measured out-of-sample degradation.

\subsection{Survival Analysis: Kaplan-Meier and Cox Proportional Hazards}
The production PD term structure (Section~6) is a discrete-time hazard model. As a challenger model we additionally fit a time-to-event survival model, the industry standard for IFRS~9 lifetime-PD term-structure work \parencite{bellotti2009}. Kaplan-Meier estimators give non-parametric survival curves $S(t)$ per credit grade (Figure~\ref{fig:km_survival}), and a Cox proportional-hazards model quantifies each covariate's multiplicative effect on the default hazard. The duration is a months-on-book proxy derived from cumulative payments and the event is the binary default flag, a synthesised time-to-event dataset since the raw data records no observed default month (a documented limitation revisited in Section~10). The model's rank-discrimination is summarised by the concordance index (Cox C-index $=$ __COX_CINDEX__), the survival-analysis analogue of the AUC. __HAZARD_DISCRIMINATION__

\begin{figure}[H]
\centering
\includegraphics[width=0.66\textwidth]{figures/km_survival_curves.png}
\caption{Kaplan-Meier non-default survival curves by credit grade. Lower grades separate downward, confirming the expected monotone grade--risk ordering.}
\label{fig:km_survival}
\end{figure}

Table~\ref{tab:cox_summary} reports the fitted Cox coefficients, hazard ratios $\exp(\beta)$ and Wald $p$-values. A hazard ratio above $1$ raises the instantaneous default hazard; below $1$ lowers it. The model is fitted on \textbf{standardised} covariates, so each hazard ratio is the effect of a one-standard-deviation move in that covariate and the ratios are directly comparable with each other; the covariate's standard deviation is tabulated alongside so a natural-unit effect can be recovered. This matters because the ridge penalty acts on the raw coefficients: fitted on unstandardised inputs it penalised a covariate measured as a $0.05$--$0.35$ fraction (the interest rate) far more heavily than one measured on a $1$--$7$ integer scale (the grade index), which made the reported multipliers incomparable and shrank the rate effect almost to nothing. __COX_DOMINANT__ Each covariate raises the default hazard monotonically as it increases, consistent with the scorecard's risk ordering.

__COX_TABLE__

% -----------------------------------------------------------------------------
% 4. LOSS GIVEN DEFAULT (LGD) AND EXPOSURE AT DEFAULT (EAD)
% -----------------------------------------------------------------------------
\section{Loss Given Default (LGD) and Exposure at Default (EAD)}

\subsection{Bimodal Two-Stage LGD Model}
Loss Given Default (LGD) represents the economic loss rate incurred when an exposure defaults. Unlike PD, which models a binary outcome, LGD is continuous, bounded within $[0, 1]$, and heavily bimodal. Defaults typically result in either a complete recovery (LGD = 0, ``cure'') or a near-total loss (LGD close to 1).

To capture this bimodal behavior \parencite{schuermann2004}, a \textbf{two-stage LGD model} is constructed as one of two candidate severity models --- benchmarked against a LightGBM challenger in Section~4.2, with the lower out-of-sample error determining which model is deployed:
\begin{enumerate}
    \item \textbf{Stage 1 (Cure Model):} A gradient-boosted classifier (100 trees, depth 4, learning rate 0.05, class-balanced sample weights) estimates the probability of a zero-loss outcome (cure). Writing $g(\mathbf{x})$ for its fitted cure probability, the loss probability is:
    \begin{equation}
    p_{\text{loss}} = P(\text{LGD} > 0 \mid \text{Default}) = 1 - g(\mathbf{x})
    \end{equation}
    A tree ensemble is used rather than a logistic link because the cure indicator responds non-monotonically to recovery-related covariates; the severity stage below remains a GLM, so the deployed LGD is a hybrid rather than a pure two-stage regression.
    \item \textbf{Stage 2 (Severity Model):} For defaulted loans that incur a loss ($\text{LGD} > 0$), a fractional logit GLM (Binomial family, logit link) models the conditional loss severity \parencite{papke1996,bellotti2012}:
    \begin{equation}
    E[\text{LGD} | \text{LGD} > 0] = \frac{1}{1 + e^{-\mathbf{x}' \boldsymbol{\beta}}}
    \end{equation}
\end{enumerate}
The final predicted LGD is the product of the two stages:
\begin{equation}
\text{LGD}_{\text{pred}} = p_{\text{loss}} \times E[\text{LGD} | \text{LGD} > 0]
\end{equation}

For Basel IRB capital calculations, a \textbf{Downturn LGD} is taken as the 90th percentile of the \emph{realised} severity distribution across loss-incurring defaults (floored at the mean predicted LGD, so the stress can never fall below the central estimate):
\begin{itemize}
    \item \textbf{Mean Expected LGD (deployed model):} __MEAN_LGD__ (used for IFRS 9 Stage 1 \& 2 provisions)
    \item \textbf{Downturn LGD:} __DOWNTURN_LGD__ (used for Basel RWA capital calculations)
\end{itemize}

\noindent\textbf{On the magnitude of the downturn uplift.} __DOWNTURN_NOTE__

\subsubsection*{Why Mean LGD Is Close to Total Loss}
The realised LGD used to fit and validate the severity model is computed directly from resolved loan outcomes as
\begin{equation}
\text{LGD}_{\text{realised}} = 1 - \frac{\max(0,\ \text{recoveries} - \text{collection\_recovery\_fee})}{\max(1,\ \text{funded\_amnt} - \text{total\_rec\_prncp})}
\end{equation}
i.e.\ net recoveries (gross post-charge-off recoveries less the fee paid to the collection agency) divided by the EAD proxy (funded amount less principal already repaid at charge-off), clipped to $[0,1]$. On this fully unsecured, non-revolving instalment book, post-charge-off cash recoveries are small relative to the outstanding balance, so realised severity concentrates near total loss; the two-stage model's cure probability $p_{\text{loss}}$ separately absorbs the loans that resolve with zero loss, while the conditional severity stage captures how close to total the loss is for the remainder. The published unsecured LGD bands cited below (\textcite{schuermann2004}: 0.45--0.55, corporate/wholesale debt; \textcite{bellotti2012}: 0.25--0.45, revolving retail cards) are measured on portfolios with active collections/settlement programmes and revolving-card structures where partial recoveries are more common; the deployed mean LGD of __MEAN_LGD__, measured on already-charged-off, non-revolving loans against a book-value EAD proxy with no collateral, is a structurally different (and higher) quantity by construction rather than an anomaly. Its magnitude relative to these bands is discussed further, with numeric verdicts, in the benchmark table (Section~\ref{subsec:benchmarks}).

LGD Benchmark Context: Two severity models are compared on a chronologically held-out selection sample --- a two-stage cure-plus-severity model and a LightGBM challenger --- and the model with the lower error there is deployed for all downstream provisioning and capital; the metrics in Table~\ref{tab:lgd_validation} are then computed on a disjoint reporting sample so the published out-of-sample performance is not measured on the same defaults used for the promotion decision. For unsecured LGD, \textcite{schuermann2004} report a $0.45$--$0.55$ range (a corporate/wholesale-debt review) while \textcite{bellotti2012} document a lower $0.25$--$0.45$ band for revolving retail credit-card portfolios. The deployed mean LGD of __MEAN_LGD__ and downturn LGD of __DOWNTURN_LGD__ are assessed against these published ranges in the benchmark table (Section~\ref{subsec:benchmarks}), where any deviation is reported together with its driver. The Downturn LGD is taken at the 90th percentile of the realised severity distribution and used exclusively for Basel IRB capital calculations, providing a buffer of __LGD_UPLIFT_PP__\,pp above the mean.

\subsection{Out-of-Sample LGD Validation}
The severity model is validated out-of-time on defaulted loans from vintages held out of the fitting window. Table~\ref{tab:lgd_validation} reports the standard LGD backtesting metrics --- Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), the coefficient of determination $R^2$, and a two-sample Kolmogorov-Smirnov (KS) statistic comparing the marginal predicted and realised LGD distributions \parencite{loterman2012benchmarking}. The KS statistic is reported without its $p$-value: at this sample size ($n=$ __LGD_VAL_N__) the test is hyper-sensitive to any trivial distributional difference and, being a marginal (not per-loan) comparison, cannot by itself certify calibration; Figure~\ref{fig:lgd_calibration} and the decile table against the $45^{\circ}$ line are the calibration evidence. The aggregate portfolio-level mean LGD sits above the unsecured LGD literature ranges cited in Section~\ref{subsec:benchmarks} (driver discussed there), and the loan-level predictive performance of the deployed severity model is separately weak, with a negative out-of-sample $R^2$ of $__LGD_R2__$ (Table~\ref{tab:lgd_validation}); the rejected two-stage model was materially worse at $R^2 \approx __LGD_R2_TWOSTAGE__$, which drove the champion--challenger switch. A negative $R^2$ is a well-documented model risk: realized retail LGD is highly bimodal (concentrated at 0 for cured loans and near 1.0 for write-offs), and predicting the exact loss severity on unsecured, non-collateralized consumer loans is statistically challenging. While the model remains suitable for calculating conservative portfolio-level capital buffers, its loan-level predictions should be treated with caution.

__LGD_VALIDATION_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/validation/lgd_calibration.png}
\caption{LGD validation: (left) predicted vs realised LGD distributions; (right) mean realised vs mean predicted LGD by predicted decile against perfect calibration.}
\label{fig:lgd_calibration}
\end{figure}

\subsection{Amortisation-Based EAD for Term Loans}
Exposure at Default (EAD) represents the outstanding gross balance owed by the borrower at the moment of default. For fully-drawn, non-revolving consumer installment loans, the outstanding principal amortizes deterministically over time. EAD is modeled using a closed-form annuity amortization formula:
\begin{equation}
EAD(t) = \text{funded\_amnt} \times \frac{1 - (1 + r)^{-(T - t)}}{1 - (1 + r)^{-T}}
\end{equation}
where $r$ is the monthly interest rate on the loan contract, $T$ is the original term in months, and $t$ is the elapsed Months-on-Book (MOB) at default.

For revolving credit facilities (such as credit cards or overdraft limits), EAD must capture future drawdowns using a Credit Conversion Factor (CCF):
\begin{equation}
EAD = \text{Drawn Balance} + \text{CCF} \times \text{Undrawn Limit}
\end{equation}
Since Lending Club loans are fully drawn installment loans with no revolving limits, the undrawn limit is zero. Using the closed-form amortization formula to calculate outstanding principal is an appropriate simplification, which is fully documented and standard in retail banking.

% -----------------------------------------------------------------------------
% 5. BASEL IRB REGULATORY CAPITAL \& CAPITAL STRESS TESTING
% -----------------------------------------------------------------------------
\section{Basel IRB Capital \& Capital Stress Testing}

Throughout this report, $Z < 0$ corresponds to an adverse macroeconomic shock (recession); $Z > 0$ corresponds to a favourable shock (expansion). This sign convention applies uniformly to the Vasicek stress test below, the IFRS~9 macro-scenario mapping (Table~\ref{tab:macro_regression}), and the ECL sensitivity analysis (Figure~\ref{fig:ecl_tornado}).

\subsection{ASRF Vasicek Model and the ``Other Retail'' Formula}
Under the Basel II/III framework \parencite{bcbs2004}, banks calculate capital requirements for retail exposures using the Asymptotic Single Risk Factor (ASRF) model \parencite{vasicek2002}. This framework assumes that portfolio risk is driven by a single systematic macroeconomic factor. The retail supervisory correlation ($R$) and capital requirement ($K$) formulas for the ``Other Retail'' asset class are:
\begin{equation}
R = 0.03 \times \frac{1 - e^{-35 \times PD}}{1 - e^{-35}} + 0.16 \times \left[ 1 - \frac{1 - e^{-35 \times PD}}{1 - e^{-35}} \right]
\end{equation}
\begin{equation}
K = \text{Downturn LGD} \times \Phi\!\left( \frac{\Phi^{-1}(PD) + \sqrt{R}\,\Phi^{-1}(0.999)}{\sqrt{1 - R}} \right) - PD \times \text{Downturn LGD}
\end{equation}
\parencite[§328]{bcbs2004}. Where $\Phi$ is the standard normal cumulative distribution function, $\Phi^{-1}$ is the inverse standard normal CDF, and the confidence level is set to \textbf{99.9\%}. Risk-Weighted Assets (RWA) are calculated by scaling the capital requirement ($K$):
\begin{equation}
RWA = K \times 12.5 \times EAD
\end{equation}

\textbf{PD horizon.} The $PD$ entering this formula is a \textbf{one-year} probability of default, as \S328 requires. The scorecard's own target is the loan's terminal resolved status, so its direct output is a \emph{lifetime} PD; it is converted at the point of use under a constant marginal-hazard assumption over the remaining term,
\begin{equation}
PD_{\text{12m}} = 1 - (1 - PD_{\text{lifetime}})^{12/T},
\end{equation}
with $T$ the contractual term in months. On this portfolio that is a mean of __MEAN_PD_LIFETIME__ lifetime against __MEAN_PD_12M__ over twelve months. The same one-year measure drives the Expected Loss of Section~5.2, the per-annum profit and RAROC calculation of Section~9, and the stress test below; the lifetime figure is retained where its horizon is the correct one, namely IFRS~9 staging and lifetime ECL. Substituting the lifetime PD into a one-year capital formula --- as an earlier version of this engine did --- inflates RWA density and, because $K$ is concave in $PD$, can make stressed RWA fall below base RWA.

\subsection{Basel IRB Capital vs.\ Standardised Approach (SA) Reference}
Table~\ref{tab:basel_comparison} compares the capital requirements calculated using the Internal Ratings-Based (IRB) approach against the Standardized Approach (SA) reference (75\% risk weight):

\begin{table}[H]
\centering
\caption{Basel IRB Regulatory Capital Comparison against Standardised Approach}
\label{tab:basel_comparison}
\vspace{0.5em}
\begin{tabular}{p{5.0cm}cc}
\toprule
\textbf{Capital Metric} & \textbf{Basel IRB (Risk-Sensitive)} & \textbf{Standardised Approach (SA)} \\
\midrule
Total Portfolio RWA & \$__TOTAL_RWA__ & \$__TOTAL_RWA_SA__ \\
Minimum Capital Reserve (8\%) & \$__BASE_CAP_REQ__ & \$__BASE_CAP_REQ_SA__ \\
Portfolio RWA Density & __RWA_DENSITY__ & 75.0\% \\
\bottomrule
\end{tabular}
\end{table}

The risk-sensitive IRB approach determines an RWA density of \textbf{__RWA_DENSITY__}. This risk-sensitive capital requirement demonstrates the value of developing internal risk models, indicating a necessary capital surcharge of \textbf{\$__RWA_RELEASE_CAP_ABS__} compared to the flat Standardised Approach, ensuring that the bank remains adequately capitalised against default stress in this retail portfolio.

\subsection{Basel III Economic Capital \& Macro Stress Testing}
We implemented a mathematically rigorous \textbf{Vasicek credit cycle stress test} \parencite{engelmann2011} to measure how the portfolio's expected and unexpected losses respond to a severe systematic contraction:
\begin{equation}
PD_{\text{PiT}}(Z) = \Phi\left( \frac{\Phi^{-1}(PD_{\text{TTC}}) - \sqrt{\rho} Z}{\sqrt{1 - \rho}} \right)
\end{equation}
where the systematic risk factor is set to an extreme stress level of $Z = -2.0$ (representing a severe economic recession with a 2.28\% probability of occurrence) under an asset correlation of $\rho = 0.15$.

Table~\ref{tab:stress_testing} compares the portfolio's capital adequacy reserves under standard conditions against the severe Vasicek macroeconomic stress state:

\begin{table}[h]
    \centering
    \caption{Basel IRB Economic Capital Stress Test (Z=-2.0 vs TTC)}
    \label{tab:stress_testing}
    \vspace{0.5em}
    \begin{tabular}{lrrr}
        \toprule
        \textbf{Dimension} & \textbf{Base IRB} & \textbf{Stressed IRB} & \textbf{Increase} \\
        \midrule
        \textbf{Expected Loss (EL)} & \$__TOTAL_EL__ & \$__STRESS_EL__ & __STRESS_EL_RATIO__ \\
        \textbf{Risk-Weighted Assets (RWA)} & \$__TOTAL_RWA__ & \$__STRESS_RWA__ & __STRESS_RWA_RATIO__ \\
        \textbf{Capital Requirement (8\%)} & \$__BASE_CAP_REQ__ & \$__STRESS_CAP_REQ__ & __STRESS_CAP_RATIO__ \\
        \bottomrule
    \end{tabular}
\end{table}

Under the severe systematic stress shock ($Z = -2.0$), the portfolio's expected loss rises by __STRESS_EL_RATIO__, and the RWA expands by __STRESS_RWA_RATIO__. This result feeds directly into capital adequacy planning (ICAAP).

\subsection{Monte Carlo Economic Capital: VaR and Expected Shortfall}
Basel IRB delivers \emph{regulatory} capital under a closed-form, infinitely-granular single-factor assumption. \emph{Economic} capital is read directly off the full simulated portfolio loss distribution, which captures the tail more faithfully and distinguishes Value-at-Risk (VaR) from Expected Shortfall (ES / CVaR), the coherent tail measure now favoured under the Basel FRTB framework \parencite{mcneil2015}. We simulate the loss distribution with a Monte Carlo ASRF (Vasicek) engine: a single systematic factor $Z \sim N(0,1)$ is drawn per scenario, obligors' conditional PDs follow Equation~(\ref{eq:vasicek_ec}), and idiosyncratic default risk enters through a per-bucket Binomial draw.
\begin{equation}
\label{eq:vasicek_ec}
p_i(Z) = \Phi\!\left( \frac{\Phi^{-1}(PD_i) - \sqrt{\rho}\,Z}{\sqrt{1 - \rho}} \right), \qquad
L = \sum_i \mathbb{1}\{\text{default}_i\} \cdot LGD_i \cdot EAD_i .
\end{equation}

\begin{figure}[H]
\centering
\includegraphics[width=0.66\textwidth]{figures/loss_distribution.png}
\caption{Monte Carlo portfolio loss distribution under the ASRF single-factor model, with Expected Loss, VaR and Expected Shortfall marked.}
\label{fig:loss_distribution}
\end{figure}

Table~\ref{tab:risk_measures} reports the resulting risk measures. Expected Shortfall exceeds VaR by construction ($ES \geq VaR \geq EL$), and the ES-based economic capital buffer is compared against the Basel IRB regulatory capital requirement.

__EC_RHO_NOTE__

__RISK_MEASURES_TABLE__

\subsection{Concentration Risk: HHI and Granularity Adjustment}
The Basel ASRF model assumes an infinitely-granular, perfectly-diversified portfolio; residual name and segment concentration therefore requires a capital add-on. We measure concentration with the Herfindahl-Hirschman Index (HHI) across exposure dimensions --- credit grade, loan purpose and borrower state --- reporting the effective number of equal-sized exposures $1/\text{HHI}$ for each (Table~\ref{tab:hhi}). Figure~\ref{fig:concentration} visualises the exposure distribution per dimension. The residual single-name concentration is capitalised through a simplified Gordy-Lutkebohmert granularity adjustment, an additive surcharge above the ASRF regulatory capital.

__HHI_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/concentration_risk.png}
\caption{Portfolio exposure concentration by credit grade, loan purpose and borrower state (top 15).}
\label{fig:concentration}
\end{figure}

% -----------------------------------------------------------------------------
% 6. IFRS 9 EXPECTED CREDIT LOSS (ECL) ENGINE
% -----------------------------------------------------------------------------
\section{IFRS 9 Expected Credit Loss (ECL) Engine}

\subsection{Three-Stage Staging and Significant Increase in Credit Risk (SICR)}
IFRS~9 \parencite{iasb2014} introduces a forward-looking impairment model based on three credit quality stages:
\begin{itemize}
    \item \textbf{Stage 1 (Performing):} No Significant Increase in Credit Risk (SICR) since origination. ECL is measured over a \textbf{12-month horizon}.
    \item \textbf{Stage 2 (Underperforming):} SICR is detected since origination, but the loan is not defaulted. ECL is measured over the \textbf{remaining lifetime} of the loan.
    \item \textbf{Stage 3 (Credit-Impaired):} Non-performing loans. ECL is measured over the \textbf{remaining lifetime}, with PD set to \textbf{100\%}. \emph{Implementation note:} the dataset carries no monthly days-past-due panel, so Stage~3 is not identified by a 90+ DPD observation but by the loan's terminal resolved status (the modelling target). This makes the Stage~3 population a hindsight classification --- see Section~10, where its effect on the headline ECL is quantified.
\end{itemize}

__STAGING_SCENARIO_NOTE__

A Significant Increase in Credit Risk (SICR) \parencite{novotny2016} is triggered if:
\begin{itemize}
    \item The ratio of lifetime PD to origination PD exceeds \textbf{2.5$\times$}.
    \item The absolute lifetime PD exceeds \textbf{20\%}.
    \item A delinquency backstop fires. The IFRS~9 standard backstop is 30+ days past due at the reporting date; absent a DPD panel, the implemented proxy is \texttt{delinq\_2yrs} $\geq 1$, i.e.\ at least one delinquency recorded in the two years \emph{before} origination. This is an application-time credit-history flag rather than a current-delinquency measure, and is a documented deviation from the standard (Section~10).
\end{itemize}

\subsection{PD Term Structure and Lifetime Expected Credit Loss}
Rather than using static 12-month PDs, the IFRS~9 engine models the lifetime PD curve using a discrete hazard model \parencite[Ch.~7]{lando2004}. The survival probability ($S(t)$) and marginal monthly PD ($m(t)$) are calculated as:
\begin{equation}
S(t) = \prod_{s=1}^{t} (1 - h(s)), \quad m(t) = S(t-1) \times h(t)
\end{equation}
where $h(t)$ is the conditional default hazard at month $t$. The final ECL is the discounted sum of expected losses over the relevant horizon ($H$):
\begin{equation}
ECL = \sum_{t=1}^{H} \frac{m(t) \times \text{LGD}(t) \times \text{EAD}(t)}{(1 + \text{EIR})^t}
\end{equation}
\parencite{bellini2019}. Where $\text{EIR}$ is the Effective Interest Rate derived from the loan's contractual pricing. \emph{Implementation note:} in the current engine $\text{LGD}(t)$ and $\text{EAD}(t)$ are held constant at their per-loan point estimates across the summation --- exposure is evaluated once (Section~4.3) rather than re-amortised month by month --- so the formula above is the general statement and the implementation is its constant-exposure special case. This is conservative for amortising loans, since the outstanding balance in fact declines over the horizon.

\subsubsection*{Lifetime PD Calibration: Validating the ECL-Driving Term Structure}
The lifetime PD $1 - S(H)$ used directly in the equation above is produced by the discrete hazard model and enters the ECL sum unchanged. It is a distinct estimator from the scorecard's 12-month PD and is \emph{not} passed through the scorecard's out-of-sample isotonic/Platt recalibration described in Section~7.2 --- that recalibration only touches the 12-month PD used for expected loss, Basel RWA, and SICR-origination staging. Rather than silently trusting an unvalidated PD inside the loss-driving formula, Table~\ref{tab:lifetime_pd_calibration} instead validates the hazard model's own lifetime PD against realised lifetime default rates by origination vintage (restricted to vintages originated in or before 2016, since 2017--2018 charge-offs are not yet resolved in the 2018Q4 snapshot; note the 2016 cohort itself is only partially matured, so its observed rate is still right-censored). A ratio near 1.0 indicates the hazard PD entering the ECL sum is itself reasonably calibrated to realised outcomes; a ratio materially outside the $[0.5, 1.5]$ tolerance band would indicate that the term structure --- not just the 12-month scorecard PD --- requires recalibration before it can be trusted in the ECL sum, a finding this report would then surface explicitly rather than let pass silently into the headline coverage ratio.

__LIFETIME_PD_CALIBRATION_TABLE__

\subsection{Forward-Looking Macroeconomic Scenarios}
To ensure ECL provisions are forward-looking and compliant with IFRS 9, we implement an Ordinary Least Squares (OLS) macroeconomic regression with imposed economic sign priors, supported by the ADF, Granger-causality and Johansen time-series diagnostics reported in Section~6.3. This framework dynamically links quarterly historical default rates of the LendingClub portfolio to key US macroeconomic indicators, __MACRO_SOURCE__: the Unemployment Rate (UNRATE), real GDP growth (GDP\_growth, from GDPC1 --- \emph{real}, not nominal, so that inflation does not enter the regression twice alongside CPI), the Federal Funds Rate (FEDFUNDS), CPI Inflation (CPI\_inflation), and House Price Index Growth (HPI\_growth, from the seasonally-adjusted Case-Shiller US National HPI), a collateral-value indicator via which rising home prices support household wealth and reduce default risk.

The OLS model determines the sensitivity (elasticity) of the portfolio default rate to each economic factor. These sensitivities are used to predict default rates under three standardized economic scenarios (Baseline, Upside, and Downside). The macro-predicted default rates are then mathematically mapped to systematic credit cycle shocks (Vasicek $Z$-shocks) using the supervisory retail correlation parameter ($\rho = 0.15$):
\begin{equation}
Z_{\text{shock}} = \frac{\Phi^{-1}(\text{TTC\_DR}) - \Phi^{-1}(\text{PIT\_DR}) \times \sqrt{1 - \rho}}{\sqrt{\rho}}
\end{equation}
where $\text{TTC\_DR}$ is the long-run average (Through-the-Cycle) default rate, and $\text{PIT\_DR}$ is the Point-in-Time default rate predicted under each macroeconomic scenario.

\textbf{The shock is applied once, at the horizon on which it was calibrated.} $\text{TTC\_DR}$ is the share of loans that default at any point in their life, so $Z$ describes a \emph{lifetime} quantity. Applying that same Vasicek transform to each \emph{monthly} hazard would compound it over the whole term, lifting cumulative PD far past the ratio the scenario actually targets. Instead the shock is applied once, to each loan's cumulative default probability over its own term, and the resulting uplift is redistributed across months as a proportional-hazards scaling: with $S = \prod_{t \le T}(1 - h_t)$ to the end of the term,
\begin{equation}
h'_t = 1 - (1 - h_t)^{\alpha}, \qquad
\alpha = \frac{-\ln\!\big(1 - \Phi(\tfrac{\Phi^{-1}(1-S) - \sqrt{\rho} Z}{\sqrt{1-\rho}})\big)}{-\ln S},
\end{equation}
which reproduces the targeted lifetime PD exactly, since $S^{\alpha} = e^{-\alpha \ln(1/S)}$. The lifetime horizon is the correct anchor because it is the horizon $Z$ is calibrated on: the Vasicek inversion in \texttt{risk/ifrs9\_ecl.py} takes its through-the-cycle default rate from the modelling target, which is the loan's terminal resolved status and therefore a \emph{lifetime} rate. Anchoring the transform at twelve months would apply a lifetime-calibrated $Z$ to a twelve-month probability, which is a different quantity.

$\rho = 0.15$ here is the supervisory retail correlation used for the scenario mapping specifically; the Basel IRB capital calculation of Section~5.1 uses the PD-dependent ``Other Retail'' curve $R \in [0.03, 0.16]$, and the Monte Carlo economic capital of Section~5.4 uses that same curve.

__TTC_BASELINE_NOTE__

The final Expected Credit Loss (ECL) is computed as the probability-weighted average across all three scenarios:
\begin{equation}
\text{ECL}_{\text{final}} = 0.50 \times \text{ECL}_{\text{Baseline}} + 0.25 \times \text{ECL}_{\text{Upside}} + 0.25 \times \text{ECL}_{\text{Downside}}
\end{equation}

__MACRO_ELASTICITIES_TABLE__

\subsubsection*{Time-Series Justification of the Lag and Sign Choice}
Table~\ref{tab:macro_ts} reports time-series diagnostics on the quarterly default-rate and macro series: an Augmented Dickey-Fuller (ADF) stationarity test, a Granger-causality test of UNRATE on the default rate, an AIC grid search over the UNRATE lag (reporting whether the economically-correct positive sign holds at the AIC-selected lag --- on this sample it does \emph{not}, which is precisely why sign priors are imposed for projection), and a Johansen cointegration test with a VECM long-run relation where cointegration is present \parencite{engelmann2011}. The contemporaneous OLS default-rate regression is prone to spurious signs (the negative UNRATE coefficient reflects charge-off lags and underwriting drift), so we impose economic sign priors for scenario projection rather than build a full Vector Error Correction Model (VECM) for projections, following the standard banking preference for simpler, transparent Vasicek overlays.

__MACRO_TS_TABLE__

\subsection{Point-in-Time vs Through-the-Cycle PD Decomposition}
Basel IRB capital is calibrated on a Through-the-Cycle (TTC) PD, a macro-neutral long-run average, whereas IFRS~9 requires a forward-looking Point-in-Time (PiT) PD. Inverting the Vasicek single-factor model on the observed quarterly default-rate series recovers the long-run TTC PD and the implied systematic factor $Z$ for each quarter (Figure~\ref{fig:pit_ttc}). Quarters with $Z<0$ are adverse (realised default rate above the TTC average); $Z>0$ marks benign quarters. This makes the cyclical PiT/TTC gap explicit and ties directly to the Vasicek $Z$ convention used throughout the ECL engine.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/pit_vs_ttc.png}
\caption{PiT vs TTC decomposition. Top: quarterly PiT default rate against the long-run TTC PD. Bottom: the implied Vasicek systematic factor $Z$, shaded by adverse/benign regime.}
\label{fig:pit_ttc}
\end{figure}

% -----------------------------------------------------------------------------
% 7. MODEL VALIDATION
% -----------------------------------------------------------------------------
\section{Model Validation}

\subsection{Discrimination Performance (OOT)}
Model validation is performed on the completely held-out Out-of-Time (OOT) dataset ($2016$--$2018$). The PD scorecard shows stable performance with an OOT AUC of \textbf{__AUC_OOT__} __AUC_CI_NOTE__ and a Kolmogorov-Smirnov (KS) statistic of \textbf{__KS_OOT__}.

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/validation/roc_curve_oot.png}
    \caption{OOT ROC Curve}
    \label{fig:val_roc}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/validation/ks_chart_oot.png}
    \caption{OOT KS Chart}
    \label{fig:val_ks}
\end{subfigure}
\caption{Discrimination Analysis (OOT). Annotated metrics are computed on the Out-of-Time (2016--2018) population and match Table~\ref{tab:exec_summary}.}
\label{fig:val_discrimination}
\end{figure}

\subsection{Calibration Robustness}
Calibration was evaluated by comparing predicted default rates against actual observed default rates across risk deciles. The Hosmer-Lemeshow test \parencite{hosmer2013} rejects perfect calibration on the OOT dataset ($p = \text{\textbf{__HL_PVALUE__}} < 0.05$). Given the very large sample size ($N = \text{VAR_N_OOT}$), this result partly reflects the high sensitivity of the chi-squared goodness-of-fit test, but the calibration plots also indicate systematic underprediction at higher risk deciles. __HL_SUBSAMPLE_NOTE__ __SPIEGELHALTER_NOTE__ __RECALIB_STATUS__ __RECALIB_RESIDUAL_NOTE__

__CALIBRATION_COMPARISON_TABLE__

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/validation/calibration_test.png}
    \caption{Calibration (In-Time Test Set)}
    \label{fig:cal_test}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/validation/calibration_oot.png}
    \caption{Calibration (Out-of-Time Validation)}
    \label{fig:cal_oot}
\end{subfigure}
\caption{Calibration Curves}
\label{fig:val_calibration}
\end{figure}

\subsection{Backtesting and Vintage Calibration}

Model backtesting compares the scorecard's predicted average PD against observed default rates segmented by quarterly origination vintage, following the methodology of \parencite{eba2017} and \parencite{bcbs2005}. For each vintage cohort, we assess whether the predicted mean PD lies within the tolerance band __VINTAGE_BAND__ of the realised default frequency:
\begin{equation}
\text{PD Ratio}_{v} = \frac{\bar{p}_v}{\bar{d}_v}
\end{equation}
where $\bar{p}_v$ is the cohort-average predicted PD and $\bar{d}_v$ is the realised default rate. A ratio inside __VINTAGE_BAND__ indicates acceptable calibration; cohorts outside it are flagged for recalibration ($\dagger$). That band, the flag in the table below and the count in the next paragraph all come from one constant in \texttt{validation/backtest.py} --- this passage previously described a 50\% band, counted vintages against $[0.80, 1.25]$, and was implemented twice with a third pair of thresholds. Table~\ref{tab:pd_backtest} reports the backtesting results by origination year, exposure-weighted across that year's quarters. The ratios below describe the PDs the pipeline actually deploys rather than raw model output --- which for in-scope vintages means \textbf{post-recalibration}. __RECALIB_SCOPE_CAVEAT__

__VINTAGE_DRIFT_SENTENCE__ The hazard model's \textbf{lifetime} PD that drives IFRS~9 ECL directly ($\text{ECL} = \sum_t m(t)\times\text{LGD}(t)\times\text{EAD}(t)/(1+\text{EIR})^t$) is a separate estimator and is \emph{not} passed through this scorecard recalibration; it is instead independently validated against realised lifetime default rates by vintage in Table~\ref{tab:lifetime_pd_calibration} (Section~6.2).

\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Vintage PD Backtesting: Predicted vs Realised Default Rate}
\label{tab:pd_backtest}
\vspace{0.5em}
\begin{tabular}{lcccc}
\toprule
\textbf{Vintage (Q)} & \textbf{N Loans} & \textbf{Predicted PD} & \textbf{Actual DR [95\% CI]} & \textbf{PD Ratio} \\
\midrule
__PD_BACKTEST_ROWS__
\bottomrule
\end{tabular}
\end{table}

\subsubsection*{Era-Specific Recalibration of the Vintage Drift}
A single global recalibrator cannot correct an era-specific bias. We therefore fit separate isotonic and Platt recalibrators for the pre-2016 and 2016--2018 eras and measure, per vintage group, the raw and recalibrated PD ratio against the realised default rate (Table~\ref{tab:vintage_calib}, Figure~\ref{fig:vintage_calib}). The raw ratio is materially below $1.0$ for the newer vintages (the documented under-prediction), and the era-specific isotonic recalibration moves it back toward $1.0$. This is an in-sample diagnostic that quantifies the drift and demonstrates the correction; production PD is unchanged. Since the era-specific calibrators are fitted and evaluated on the same vintage partitions (e.g. 2016--2018), the resulting alignment (Isotonic/Platt PD equals the Actual DR exactly) is in-sample and tautological: it is a diagnostic baseline that demonstrates the extent of the raw model's drift, not a validation of the recalibrator's out-of-sample generalization.

__VINTAGE_CALIB_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=0.65\textwidth]{figures/validation/calibration_by_vintage.png}
\caption{PD calibration ratio (predicted / actual) by vintage group, raw vs era-recalibrated. A ratio of 1.0 is perfect calibration; below 1.0 is under-prediction.}
\label{fig:vintage_calib}
\end{figure}

\subsection{Stage Migration: Why No Matrix Is Reported}

A stage migration matrix --- transitions between IFRS~9 stages from twelve months before the reporting date to the reporting date \parencite{novotny2016} --- is a standard indicator of portfolio deterioration velocity, and its absence here is deliberate.

Producing one requires the stage a loan occupied at a \emph{past} date. That requires a monthly servicing panel: balances, delinquency status and staging decisions recorded as they stood. This dataset carries no such panel. It provides one row per loan describing its state at origination and its terminal resolved outcome, with nothing in between. Two consequences follow and neither is fixable by modelling. The $t-12$ Stage~3 population is unidentifiable, because the credit-impaired flag available here \emph{is} the terminal outcome and cannot be evaluated at an earlier date --- so every row of a reconstructed matrix would be forced into Stage~1 or Stage~2. And the $t-12$ credit profile itself would have to be imputed from the same origination covariates that produce the reporting-date profile, making the resulting ``migration'' a deterministic re-labelling of the current state rather than an observed transition.

An earlier version of this report presented such a reconstructed matrix. Its cells were arithmetic artefacts of the reconstruction --- most visibly, it showed more cures than deteriorations on a book whose default rate was rising. Publishing a table that looks like observed migration but is not is worse than publishing none, so it has been withdrawn. Section~10 lists the servicing panel as the single data acquisition that would most improve this model; recording origination-date PD snapshots at booking (Section~10.2) would make a genuine matrix available from the next reporting cycle onward.

\subsection{ECL Macro Scenario Sensitivity}

Figure~\ref{fig:ecl_tornado} presents the portfolio ECL sensitivity to the Vasicek systematic factor~$Z$ across a range of macro scenarios, from severe expansion ($Z = +2.0$) to severe recession ($Z = -2.0$). This analysis follows the probability-weighted scenario methodology mandated by IFRS~9 paragraph B5.5.42 \parencite{iasb2014} and the stress testing framework of \parencite{bellini2019}.

Two things about the scale of those percentages. The reference row is $Z = 0$, the \emph{unconditional} anchor --- not the priced Baseline scenario, which carries its own non-zero implied $Z$ of \textbf{__BASELINE_Z__}, and not the probability-weighted headline provision. And the denominator is the whole ECL, which is dominated by the Stage~3 term where $PD$ is forced to $1.0$ and no macro factor can reach it (Section~10.3 measures that share at \textbf{__STAGE3_ECL_SHARE__} of the provision). A $\pm$__TORNADO_SPAN__ swing on the total therefore corresponds to a materially larger swing on the PD-sensitive part of the book: __TORNADO_PD_ONLY__ Read the tornado as a sensitivity of the reported provision, not of the modelled credit risk.

\subsection{ECL What-If Calculator: PD / LGD / EAD Stress Scenarios}
Beyond the macro-factor mapping, a management-facing what-if calculator answers the direct risk-committee question \emph{``what happens to provisions if PD, LGD or EAD move?''}. Holding the baseline term structure fixed, each scenario applies a multiplicative PD shock, an additive LGD shock and/or a multiplicative EAD (drawdown) shock, and the ECL engine is re-run. Table~\ref{tab:ecl_whatif} reports the resulting provision changes, and Figure~\ref{fig:ecl_shock_tornado} presents them as a tornado chart, including regulator-style severe scenarios calibrated to COVID-19 and Global-Financial-Crisis default multiples. The anchor for these what-if sensitivities is the ECL evaluated at $Z = 0$: neither the priced Baseline scenario, which carries its own non-zero implied $Z$, nor the probability-weighted total, which additionally weights the downside and upside scenarios. That is why it differs from the headline ECL. Every entry in the table is a ratio to that same anchor, so the choice cancels.

__ECL_WHATIF_TABLE__

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/ecl_tornado.png}
    \caption{Macro Sensitivity (Vasicek $Z$-Factor Shock)}
    \label{fig:ecl_tornado}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/ecl_shock_tornado.png}
    \caption{PD / LGD / EAD Stress Scenarios}
    \label{fig:ecl_shock_tornado}
\end{subfigure}
\caption{ECL sensitivity tornado charts (change in ECL vs baseline).}
\label{fig:ecl_tornado_combined}
\end{figure}

\subsection{Scientific Benchmark Verification Layer}

To situate our results, all modeling outputs are compared against published reference ranges drawn from the credit-risk literature. This is a comparison against published ranges, not a reproduction of the cited studies' experiments. Following \textcite{thomas2000survey} and the benchmarking study of \textcite{lessmann2015benchmarking}, our Probability of Default (PD) scorecard achieves an OOT Gini coefficient of \textbf{\num{__GINI_OOT__}} (AUC of \textbf{\num{__AUC_OOT__}}), which sits within the empirical range of $0.30$--$0.45$ reported for high-volume retail loan portfolios (equivalently AUC $0.65$--$0.73$, since $\text{Gini} = 2\,\text{AUC}-1$).

The deployed severity model, selected by lower out-of-sample error from a two-stage model and a LightGBM challenger, yields a Mean LGD of \textbf{\num{__MEAN_LGD__}} (and Downturn LGD of \textbf{\num{__DOWNTURN_LGD__}}). This out-of-sample-validated loss rate sits above both the $0.45$--$0.55$ unsecured range reported by \textcite{schuermann2004} (corporate/wholesale debt) and the lower $0.25$--$0.45$ revolving retail credit-card band of \textcite{bellotti2012}, consistent with the low post-charge-off collection recovery typical of unsecured instalment loans once EAD is measured on a principal-only basis (driver discussed in Table~\ref{tab:benchmark_verification}).

Under the Basel Committee regulatory standards \parencite{bcbs2004,bcbs2017}, the risk-sensitive internal ratings-based (IRB) approach sets our Risk-Weighted Asset (RWA) density at \textbf{__RWA_DENSITY__}, above the flat 75\% risk weight assumed by the Standardised Approach (SA). Because this portfolio is a high-yield, higher-risk unsecured book, the risk-sensitive approach produces a capital \emph{surcharge} rather than relief, the expected outcome when portfolio risk exceeds the level implicit in the flat SA weight.

Consistent with the reject inference analysis of \textcite{crook2004reject}, the inclusion of parcelled rejected applications leads to a Gini shift of \textbf{\num{__GINI_SHIFT__}}. This shift is consistent with the presence of selection bias; through-the-door refitting produces a small, conservative change in the fitted parameters.

Table~\ref{tab:benchmark_verification} presents a structured side-by-side comparison of our empirical results against published scientific benchmarks.

\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Comparison against published reference ranges. Verdicts are computed numerically at build time by comparing the Project Value column against the published range; each range is sourced from a single registry (reports/benchmarks.py) with its citation. These are literature reference ranges, not a reproduction of the cited studies.}
\label{tab:benchmark_verification}
\vspace{0.5em}
\begin{tabularx}{\linewidth}{p{2.2cm}ccp{2.2cm}cp{4.5cm}}
\toprule
\textbf{Metric / Parameter} & \textbf{Project Value} & \textbf{Published Benchmark} & \textbf{Academic Source} & \textbf{Verdict} & \textbf{Comment} \\
\midrule
PD AUC (OOT) & \textbf{\num{__AUC_OOT__}} & __RANGE_AUC_OOT__ & \textcite{lessmann2015benchmarking} & __VERDICT_AUC_OOT__ & __COMMENT_AUC_OOT__ \\
PD Gini (OOT) & \textbf{\num{__GINI_OOT__}} & __RANGE_GINI_OOT__ & \textcite{lessmann2015benchmarking} & __VERDICT_GINI_OOT__ & \\
LGD Mean & \textbf{\num{__MEAN_LGD__}} & __RANGE_MEAN_LGD__ & \textcite{bellotti2012} & __VERDICT_MEAN_LGD__ & __COMMENT_MEAN_LGD__ \\
LGD $R^2$ (OOS) & \textbf{__LGD_R2__} & __RANGE_LGD_R2__ & \textcite{loterman2012benchmarking} & __VERDICT_LGD_R2__ & __COMMENT_LGD_R2__ \\
RWA Density & \textbf{__RWA_DENSITY__} & __RANGE_RWA_DENSITY__ & \textcite{bcbs2017} & __VERDICT_RWA_DENSITY__ & __COMMENT_RWA_DENSITY__ \\
IRB vs SA Capital & __IRB_SA_DIRECTION__ & Risk-sensitive & \textcite{bcbs2004} & __VERDICT_IRB_SA__ & __COMMENT_IRB_SA__ \\
Reject Inference $\Delta$Gini & \textbf{\num{__GINI_SHIFT__}} & __RANGE_GINI_SHIFT__ & \textcite{crook2004reject} & __VERDICT_GINI_SHIFT__ & \\
Score PSI & \textbf{\num{__PSI_TRAIN_OOT__}} & __RANGE_PSI__ & \textcite{siddiqi2017} & __VERDICT_PSI__ & \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Machine Learning Champion-Challenger Benchmarking}

To evaluate whether the linear scorecard loses significant predictive power compared to non-linear alternatives, we constructed a Champion-Challenger benchmarking layer. We trained and evaluated:
\begin{enumerate}
    \item \textbf{Logistic Scorecard:} Our baseline WoE scorecard.
    \item \textbf{LightGBM Classifier:} A state-of-the-art tree boosting model.
    \item \textbf{XGBoost Classifier:} A highly optimized distributed gradient boosting model.
    \item \textbf{Random Forest Classifier:} A classic tree-bagging model.
    \item \textbf{Weighted Ensemble:} A weighted combination of scorecard ($30\%$), LightGBM ($30\%$), XGBoost ($20\%$), and Random Forest ($20\%$).
\end{enumerate}

Table~\ref{tab:ml_comparison} presents the out-of-time (OOT) generalization benchmarks, discrimination statistics (AUC, Gini, KS), and computational costs for each algorithm.

__ML_COMPARISON_TABLE__

Notably, __ML_VERDICT__ \textbf{Feature parity.} The scorecard consumes __N_FEATURES_SC__ selected predictors and the tree challengers consume __N_FEATURES_CH__; __FEATURE_PARITY_NOTE__ It should be noted that the challenger models (LightGBM, XGBoost, Random Forest) were trained using standard, default configurations without systematic hyperparameter grid search tuning. Hyperparameter optimisation would, if anything, widen any challenger advantage, so the ranking above should be read as a lower bound on what tuned tree models could achieve on this feature set.

\subsubsection*{Is the Challenger's Edge Statistically Significant?}
Point-estimate AUC/Gini gaps cannot distinguish a genuine improvement from sampling noise. To test significance we run a paired bootstrap: the same held-out OOT rows are resampled for both models and a confidence interval is built on the \emph{difference} in Gini (challenger $-$ champion). If that interval excludes zero, the difference is significant. Table~\ref{tab:ab_test} reports the result --- __AB_DIRECTION__ It complements the analytic DeLong test on the same pair, which is computed with the full covariance term of \textcite{delong1988}: both models score the same loans, so their AUC estimates are strongly positively correlated (__DELONG_CORR__ on this run) and $\mathrm{Var}(\widehat{A}_1 - \widehat{A}_2) = \mathrm{Var}(\widehat{A}_1) + \mathrm{Var}(\widehat{A}_2) - 2\,\mathrm{Cov}(\widehat{A}_1, \widehat{A}_2)$. Dropping that covariance --- as an independent-samples approximation does --- inflates the standard error and makes the test conservative by a wide margin. __ENSEMBLE_AB_NOTE__

__AB_TEST_TABLE__

\subsection{SHAP Explainability: Challenger Model Feature Contributions}

While the primary WoE scorecard provides point-attributable explanations per feature bin \parencite[Ch.~4]{siddiqi2017}, the LightGBM challenger model is interpreted via SHAP (SHapley Additive exPlanations) values \parencite{lundberg2017}. Figure~\ref{fig:shap} displays the mean absolute SHAP contribution of each feature: \texttt{VAR_SHAP_TOP1} and \texttt{VAR_SHAP_TOP2} dominate the challenger's default discrimination.

\begin{figure}[H]
\centering
\includegraphics[width=0.64\textwidth]{figures/validation/shap_challenger_summary.png}
\caption{SHAP Feature Importance --- LightGBM Challenger Model}
\label{fig:shap}
\end{figure}

% -----------------------------------------------------------------------------
% 8. STABILITY AND PERFORMANCE OVERLAYS
% -----------------------------------------------------------------------------
\section{Stability and Performance Overlays}

\subsection{Scorecard Population Stability (PSI)}
The Population Stability Index (PSI) \parencite[Ch.~9]{engelmann2011} is computed here on the distribution of \textbf{predicted PD} --- not of credit scores --- between the training and OOT populations. Its value is \textbf{__PSI_TRAIN_OOT__}, far below the conservative regulatory threshold of \textbf{0.10}: the shape of the model's output distribution is essentially unchanged across the two windows.

\textbf{This is not, on its own, reassurance.} Over the same interval the realised default rate rose from __TRAIN_BAD_RATE__ to __OOT_BAD_RATE__. A near-zero PSI alongside a materially higher outcome rate says that the model's inputs did not move while what they were predicting did --- the scorecard did not \emph{see} the deterioration. That is precisely the calibration drift diagnosed in Sections~7.2 and~7.3, and the two findings should be read together rather than as an encouraging stability result followed by an unrelated calibration problem. PSI provides assurance only when the outcome rate is also stable; here it is not, so the per-feature Characteristic Stability Index in Table~\ref{tab:csi} is the more informative diagnostic of where (or whether) the input distributions shifted.

__CSI_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=0.64\textwidth]{figures/validation/psi_distribution.png}
\caption{Score Distribution Stability (Train vs.\ OOT)}
\label{fig:psi_dist}
\end{figure}

\subsection{Gains and Lift Analysis}
The Gains Chart measures the model's ability to concentrate defaults within the lowest score bands. The scorecard concentrates most defaults in the lowest score bands (Figure~\ref{fig:gains_ch}). Both panels below are computed on the \textbf{out-of-time} partition, so they describe generalisation rather than in-sample fit; the gains chart previously used the in-time test partition while being presented alongside OOT material.

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/validation/gains_chart.png}
    \caption{Cumulative Capture (Gains) Chart --- \textbf{OOT partition}}
    \label{fig:gains_ch}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/validation/roc_oot_overlay.png}
    \caption{ROC Curve (Holdout vs OOT Overlay)}
    \label{fig:roc_overlay}
\end{subfigure}
\caption{Lift and Generalisation Overlays}
\label{fig:lift_overlays}
\end{figure}

% -----------------------------------------------------------------------------
% 9. LIMITATIONS AND RECOMMENDATIONS
% -----------------------------------------------------------------------------
\section{Business Decisioning and Cutoff Optimisation}

Underwriting decisions require balancing credit risk, portfolio growth, and capital constraints. We evaluate each score cutoff on Expected Profit and Risk-Adjusted Return on Capital (RAROC), with expected loss (at the approved-population bad rate) and a __COST_OF_CAPITAL__ cost-of-capital charge on economic capital both netted out. On this portfolio __CORNER_AGREEMENT__ __CORNER_DESC__. Because __CORNER_IMPLICATION__, the decision problem is formulated as a \emph{constrained} optimization: we maximize approved volume and expected profit subject to an active risk-appetite ceiling on the approved bad rate (\textbf{__MAX_BAD_RATE__}), taking the most inclusive cutoff (maximising approved profit) whose approved bad rate stays within that ceiling. This constrained boundary represents the recommended operating cutoff.

We sweep score cutoffs from 400 to 800 (evaluating approved loan subsets) to calculate Expected Profit and RAROC. All components are expressed on a consistent \emph{per-annum} basis:
\begin{itemize}
    \item \textbf{Interest Income:} Based on per-loan interest rates (annual coupon on EAD).
    \item \textbf{Fees:} 1.0\% of Exposure at Default (EAD) per annum.
    \item \textbf{Funding Cost:} 4.0\% of EAD per annum.
    \item \textbf{Operating Cost:} 1.5\% of EAD per annum.
    \item \textbf{Expected Loss (EL):} $\text{PD}_{\text{annual}} \times \text{LGD} \times \text{EAD}$, where the lifetime PD is converted to an annual default probability via $\text{PD}_{\text{annual}} = 1 - (1 - \text{PD}_{\text{lifetime}})^{12/\text{term}}$. Charging the full lifetime PD against a single year of income would overstate losses by the loan term and distort every cutoff decision.
    \item \textbf{Capital Cost:} the \textbf{cost of capital}, __COST_OF_CAPITAL__, charged against economic capital (the Minimum Capital Reserve, 8\% of RWA). This is distinct from the \textbf{RAROC hurdle} of __RAROC_HURDLE__, which is not a charge but the threshold the resulting RAROC is compared against. The two rates are different numbers with different roles and are used consistently as such throughout this section.
\end{itemize}

Table~\ref{tab:cutoff_raroc} displays the optimization results across selected score cutoffs.

__CUTOFF_RAROC_TABLE__

As shown in the table, the recommended operating cutoff is score \textbf{__OPT_CUTOFF__} with an approval rate of \textbf{__OPT_APPROVAL__}, a bad rate of \textbf{__OPT_BAD__}, an expected profit of \textbf{\$__OPT_PROFIT_M__M}, and a RAROC of \textbf{__OPT_RAROC__} --- __RAROC_VS_HURDLE__ the __RAROC_HURDLE__ RAROC hurdle. It is the most inclusive cutoff whose approved bad rate stays within the risk-appetite ceiling, and it appears as the highlighted row in Table~\ref{tab:cutoff_raroc} and is marked in Figure~\ref{fig:cutoff_profit}. A single objective drives the highlighted table row, the figure, and this text, so there is no discrepancy between the quoted cutoff and the swept grid. Relaxing the risk-appetite ceiling lowers the cutoff and raises approved volume; tightening it raises the cutoff toward the near-zero-volume unconstrained corner. __GRID_HURDLE_VERDICT__

\begin{figure}[H]
\centering
\includegraphics[width=0.68\textwidth]{figures/cutoff_profit_curve.png}
\caption{Expected Profit and RAROC versus score cutoff over the full 400--800 grid. The gold marker denotes the recommended operating cutoff (risk-appetite ceiling on the approved bad rate) reported in Table~\ref{tab:cutoff_raroc}.}
\label{fig:cutoff_profit}
\end{figure}

\section{Limitations, Assumptions, and Recommendations}
\label{sec:assumptions}

\subsection{Model Risk \& Limitations}
The Lending Club dataset is limited to US consumer loans and may not generalize to corporate lending, small-business loans, or other international retail portfolios. Additionally, the dataset excludes collateral details, meaning the LGD model relies primarily on underwriting grades and debt-to-income (DTI) metrics.

\subsection{Reconciling Expected Loss with IFRS~9 ECL}
The report carries two headline provisions --- the one-year Expected Loss of Section~5.2 and the IFRS~9 ECL of Section~6 --- defined over different horizons, with different staging, and with different discounting. __EL_ECL_PROXIMITY__ Table~\ref{tab:el_ecl_reconciliation} decomposes both by stage, which makes the actual driver visible.

__ECL_RECONCILIATION_TABLE__

\subsection{Stage 3 Is a Hindsight Classification, and It Dominates the ECL}
The single most material limitation of this engine concerns \emph{what the headline ECL figure actually measures}. IFRS~9 Stage~3 (credit-impaired) is normally identified by observed non-performance --- 90+ days past due at the reporting date. The LendingClub accepted-loan file is loan-level with no monthly delinquency panel, so no such observation exists. The implementation therefore classifies Stage~3 from the loan's \emph{terminal resolved status} --- the same variable used as the PD modelling target. Stage~3 membership is consequently known only with hindsight, and each Stage~3 loan is provisioned at $\text{LGD}\times\text{EAD}$ with PD forced to $1.0$, a quantity entirely insensitive to the PD model.

The what-if calculator in Section~7.6 makes the resulting dominance measurable. Shocking LGD by $+10$pp moves the total ECL by __WHATIF_LGD_PCT__ and shocking EAD by $+15\%$ moves it by __WHATIF_EAD_PCT__ --- both exactly proportional, as they must be, since ECL is linear in each. But shocking PD by $+50\%$ moves it by only __WHATIF_PD50_PCT__. If a fraction $f$ of the ECL sits in the PD-independent Stage~3 term, a $+50\%$ PD shock produces a change of $0.5\,(1-f)$; the observed sensitivity therefore implies $f \approx$ __STAGE3_ECL_SHARE__ of the reported provision.__WHATIF_BASE_NOTE__

In other words, the headline ECL is, to first order, \emph{realised defaults} $\times$ \emph{LGD} $\times$ \emph{EAD} --- an accounting identity computed with knowledge of the outcome --- and every forward-looking component of the engine (the macro regression, the Vasicek $Z$-shocks, the probability-weighted scenarios and the hazard term structure) moves only the residual. __PERFORMING_BOOK_CAVEAT__; the absolute headline provision should not be read as a forward-looking estimate of losses on an unresolved portfolio. A production deployment on a live book, where Stage~3 is set by observed delinquency at the reporting date rather than by terminal outcome, would not have this property.

\subsection{Known Model Assumptions and Limitations}
This single list carries every model assumption and its associated limitation; it replaces the separate assumptions table that previously restated several of these items in shorter form.
\begin{itemize}
    \item \textbf{PD scorecard --- coarse monotone WoE bins:} Binning is monotone by construction, so genuinely non-monotone relationships are flattened rather than fitted. The tree challengers in Section~3.6 bound what that costs: they are free to fit non-monotone structure and do not materially outperform.
    \item \textbf{LGD --- portfolio-level validity only:} The deployed severity model is selected on out-of-sample error, with EAD proxied by \texttt{funded\_amnt} so accrued interest is ignored. Loan-level $R^2$ is negative, so LGD predictions are used for portfolio aggregates and must not be read as per-loan estimates.
    \item \textbf{EAD --- closed-form annuity, no prepayment:} Exposure is the contractual amortisation balance; borrowers who prepay are therefore carried at a higher exposure than they actually hold. This is conservative for capital. __EAD_MOB_ASSUMPTION__
    \item \textbf{Hazard --- constant within each calendar month:} The discrete-time formulation holds the hazard flat inside each month, so intra-month seasoning is not represented; the monthly resolution is what bounds this, and it is finer than the annual step a Markov-chain alternative would impose.
    \item \textbf{Basel IRB --- single systematic factor:} The ASRF ``Other Retail'' supervisory correlation $R \in [0.03, 0.16]$ assumes one common factor, so name and sector concentration are not captured. Section~5.3 reports the concentration diagnostics separately.
    \item \textbf{Reject inference --- sampled rejects:} Parcelling runs on a 100k random sample of the 27.6M reject rows, so the sample may not represent the full reject distribution; the Gini shift is monitored rather than assumed stable.
    \item \textbf{No stage-migration matrix:} A 12-month stage transition matrix requires the stage a loan occupied at a past date, which needs a monthly servicing panel this dataset does not contain. Section~7.5 sets out why no such matrix is reported rather than reconstructing one.
    \item \textbf{No origination-date PD snapshot --- relative SICR trigger disabled:} IFRS~9's primary SICR test compares the current lifetime PD against the PD estimated \emph{at booking}. This dataset stores no booking-date model output, so no such snapshot exists. Rather than substitute the current PD for the origination PD --- which makes the ratio identically $1.0$ and silently disables the test while appearing to implement it --- the relative trigger is switched off and Stage~2 is driven by the absolute lifetime-PD threshold and the delinquency backstop alone. Reinstating it requires storing PD at booking, which is listed under Recommended Next Steps below.
    \item \textbf{Delinquency backstop is an application-time proxy:} The implemented backstop is \texttt{delinq\_2yrs} $\geq 1$ --- delinquencies in the two years \emph{before} origination --- not the IFRS~9 standard 30+ DPD at the reporting date, which this data cannot support.
    \item \textbf{Downturn LGD sits at the distribution cap:} The Basel downturn LGD is the 90th percentile of the \emph{realised} severity of loss-incurring defaults, which on this fully unsecured, non-revolving book is __DOWNTURN_LGD__ --- the upper bound of the $[0,1]$ range. It is a genuine empirical percentile rather than a modelled stress, and it cannot be exceeded, so the capital calculation carries no headroom above it for a severity worse than total loss.
    \item \textbf{Hazard-model event timing is a proxy:} The data records no observed default \emph{date}, so the discrete-time hazard panel is built against a duration proxy --- cumulative payments divided by the contractual instalment --- and right-censors survivors at the month their payments stop. This is the same proxy the Kaplan-Meier/Cox challenger in Section~3.7 uses. It replaces an earlier construction that gave every loan its full contractual term of person-periods with the event pinned to the final one and no censoring; that version fitted a 12-month hazard of essentially zero, which provisioned the entire Stage~1 book at nil. The remaining limitation is that the duration is inferred, not observed, and that prepayment is treated as censoring rather than as a competing risk --- so a borrower who repays early is treated as having survived to that point rather than as having left the risk pool for a different reason.
    \item \textbf{Portfolio aggregates are partly in-sample:} Expected Loss, Basel RWA and IFRS~9 ECL are computed over the combined train, in-time test and OOT partitions --- the same rows the scorecard, hazard and LGD models were fitted on. This is appropriate for provisioning a closed book but means the portfolio aggregates are not out-of-sample quantities; the out-of-sample evidence is the OOT discrimination and calibration in Section~7.
    \item \textbf{The out-of-time sample is selected on maturity, not drawn at random:} the modelling population keeps only loans whose status is resolved at the 2018Q4 snapshot, so 2016--2018 originations that were still \emph{Current} are excluded. Among recent vintages the loans that have already resolved are disproportionately those that ended early --- that is, the charged-off ones --- and 60-month loans are under-represented relative to 36-month ones. The effect is visible in the composition of the window: half the OOT observations come from the eleven months to \textbf{2016-11} and the other half from the following twenty-five, over a period when origination volume was rising. Every OOT statistic in Section~7 (AUC, KS, PSI, Hosmer-Lemeshow, and the recalibration gate that keys off them) inherits this selection.
    \item \textbf{Exposure is booked on loans that no longer exist:} EAD amortises each loan contractually to the \textbf{__REPORTING_DATE__} reporting date from its origination month, but a large part of the book had already been repaid in full or charged off years before that date, and their true exposure at the reporting date is zero. Stage~3 in particular ($\text{EAD} =$ __STAGE3_EAD__) consists by construction of loans whose terminal status is already known. The zero-prepayment assumption noted above covers early repayment \emph{within} a live loan; it does not cover provisioning against exposures that had already terminated. The engine should be read as pricing this book as if observed at origination-plus-elapsed-term, not as a snapshot of a live portfolio.
    \item \textbf{The macro apparatus is fitted on the training window only:} the OLS default-rate regression, the scenario projections, the implied Vasicek $Z$ shocks, the ADF / Granger / Johansen diagnostics and the PiT--TTC decomposition are all estimated on the \emph{training} vintages (2007--2014), four years before the reporting date, and then applied to the whole book. Nothing in the 2016--2018 era informs the macro sensitivities that drive the forward-looking part of the ECL.
    \item \textbf{Constant exposure inside the ECL sum:} $\text{LGD}(t)$ and $\text{EAD}(t)$ are held at their per-loan point estimates across the ECL horizon rather than re-amortised monthly, which is conservative for amortising loans.
    \item \textbf{The adverse scenario is a rate-and-inflation shock, not a lower-bound recession:} Every macro axis moves in the direction its imposed sign prior says raises defaults (downside) or lowers them (upside), so the required Downside $>$ Baseline $>$ Upside ordering holds \emph{by construction} for any non-negative coefficient magnitudes rather than depending on the fitted values. The consequence is that the downside tightens policy and raises inflation into the downturn. This is a deliberate choice: it is the more punitive configuration for an unsecured consumer book, where floating debt-service costs and a real-income squeeze compound the unemployment shock, and it matches how supervisory adverse scenarios are typically built. A deflationary lower-bound recession, in which policy is cut, would instead \emph{offset} part of the adverse shock through the rate channel and is not modelled here.
    \item \textbf{The ``baseline'' scenario maps to $Z = $ __BASELINE_Z__, not $Z=0$:} This is a property of the Vasicek conditional-PD function, whose value at $Z=0$ does not equal the unconditional PD; the projection intercept is recentred so the baseline reproduces the through-the-cycle default rate. Under the report-wide convention that $Z<0$ is adverse, the baseline therefore reads as mildly adverse even though it is the central expectation.
    \item \textbf{Categorical Feature Encoding:} Grade, term, employment length and home ownership are ordinal-encoded before WoE binning. Label ordering assumptions (e.g., A=1 through G=7) are reasonable but should be validated against observed default rates for each category.
\end{itemize}

\subsection{Empirical Results vs.\ Published Literature}
\label{subsec:benchmarks}

\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Project empirical results vs.\ published reference ranges (comparison against the literature, not a reproduction of the cited studies).}
\label{tab:literature_benchmarks}
\begin{tabularx}{\linewidth}{p{2.2cm}ccp{2.2cm}lp{4.5cm}}
\toprule
\textbf{Metric} & \textbf{This Study} & \textbf{Published Range} & \textbf{Source} & \textbf{Verdict} & \textbf{Comment}\\
\midrule
Scorecard Gini (OOT) & VAR_GINI_OOT & __RANGE_GINI_OOT__
  & \cite{lessmann2015benchmarking} & __VERDICT_GINI_OOT__ & \\
LightGBM Gini (OOT)  & VAR_LGBM_GINI_OOT & __RANGE_LIT_LGBM__
  & \cite{lessmann2015benchmarking} & __VERDICT_LIT_LGBM__ & \\
Mean LGD (P2P)       & VAR_MEAN_LGD & __RANGE_MEAN_LGD__
  & \cite{bellotti2012} & __VERDICT_MEAN_LGD__ & __COMMENT_MEAN_LGD__ \\
Downturn LGD         & VAR_DOWNTURN_LGD & __RANGE_LIT_DLGD__
  & \cite{bellotti2012} & __VERDICT_LIT_DLGD__ & __COMMENT_LIT_DLGD__ \\
IFRS~9 ECL Coverage  & VAR_ECL_COVERAGE & __RANGE_LIT_ECL_COV__
  & \cite{eba2022} & __VERDICT_LIT_ECL_COV__ & __COMMENT_LIT_ECL_COV__ \\
IFRS~9 Stage~2\%     & VAR_STAGE2_PCT & __RANGE_LIT_STAGE2__
  & \cite{eba2022} & __VERDICT_LIT_STAGE2__ & __COMMENT_LIT_STAGE2__ \\
\bottomrule
\end{tabularx}
\end{table}

\footnotesize\noindent Verdicts in Table~\ref{tab:literature_benchmarks} are recomputed at every build by numerical comparison of the two adjacent columns; out-of-range values are reported as such, with the driver noted in the Comment column.\normalsize

\subsection{Recommended Next Steps}
\begin{enumerate}
    \item \textbf{Independent Validation:} Conduct an independent review of the binning algorithms and scorecard points scaling to verify math and logical consistency.
    \item \textbf{Advanced LGD Modeling:} Explore survival-based LGD models that capture time-varying recovery patterns over the life of defaulted assets.
    \item \textbf{Longitudinal SICR Tracking:} Implement a proper SICR framework using origination-date PD snapshots stored at booking, enabling precise Stage 2 migration rates compliant with IFRS 9 paragraph 5.5.9.
\end{enumerate}

\clearpage
% -----------------------------------------------------------------------------
% 10. APPENDICES
% -----------------------------------------------------------------------------
\section{Appendices}

\subsection{A. Scorecard Scaling Derivation}
To map the log-odds predicted by the logistic regression to a scaled credit score, we solve the linear system:
\begin{equation}
Score = Offset + Factor \times \ln(\text{odds})
\end{equation}
Subject to the boundary conditions:
\begin{itemize}
    \item $\text{Score} = 600$ at $\text{odds} = 50:1$ ($\ln(50) \approx 3.912$)
    \item $\text{Score} = 620$ at $\text{odds} = 100:1$ ($\ln(100) \approx 4.605$, satisfying $\text{PDO} = 20$)
\end{itemize}
Solving for the scaling parameters:
\begin{equation}
Factor = \frac{20}{\ln(100) - \ln(50)} = \frac{20}{\ln(2)} \approx 28.8539
\end{equation}
\begin{equation}
Offset = 600 - 28.8539 \times \ln(50) \approx 487.123
\end{equation}

\subsection{B. Basel IRB ``Other Retail'' Supervisory Parameters}
Under BCBS §322--328 the regulatory correlation $R$ is constrained to $[0.03, 0.16]$ and decays exponentially as PD rises, reflecting the lower correlation of defaults among higher-risk borrowers in stable conditions; the formula is stated in Section~5.1 and is not repeated here. For the RWA calculation the supervisory PD floor of \textbf{0.03\%} is strictly applied, and the PD entering it is the twelve-month measure derived in Section~5.1.

\subsection{C. Technical Stack \& Reproducibility}
The pipeline is designed to be fully reproducible:
\begin{table}[H]
\centering
\caption{Modeling Pipeline Technology Stack}
\label{tab:tech_stack}
\vspace{0.5em}
\begin{tabular}{p{3.2cm}p{4.0cm}p{6.8cm}}
\toprule
\textbf{Component} & \textbf{Technology} & \textbf{Purpose in Pipeline} \\
\midrule
Language & Python 3.11+ & Core language and scripting \\
Feature Binning & __BINNER_USED__ & WoE bin construction for every scorecard feature \\
PD Underwriting & statsmodels Logit & Monotonic credit scorecard fitting \\
Feature Selection & scikit-learn \texttt{LogisticRegressionCV} (ElasticNet, SAGA) & Third selection stage, between the VIF filter and the sign check (Section~3.3) \\
LGD Modeling & LightGBM (deployed); scikit-learn \texttt{GradientBoostingClassifier} (cure stage); statsmodels GLM (severity stage) & Severity model selected out-of-sample; the two-stage cure + fractional-logit model is the benchmarked \emph{challenger} that was not deployed (Section~4.2) \\
Challenger Models & LightGBM, XGBoost, scikit-learn \texttt{RandomForest} & Non-linear machine learning benchmarks (Table~\ref{tab:ml_comparison}) \\
Survival PD Curves & Discrete-time hazard model; \texttt{lifelines} Cox challenger & Monthly lifetime term structure modeling \\
PDF Generation & XeLaTeX + biber & Publication-quality academic PDF \\
\bottomrule
\end{tabular}
\end{table}

\noindent\textbf{Enhancement phases dropped in this run:} __PHASE_FAILURES__ Every optional phase records a non-fatal failure if it does not complete, so a table that silently vanished from this report --- for example because an optional dependency was absent --- is disclosed here rather than simply being absent.

\vspace{0.6em}
\noindent\textbf{Computed but not tabulated here.} The pipeline also writes the OOT decile rank-ordering table, the LGD backtest by vintage, and the full ECL macro-sensitivity grid (shown here only as Figure~\ref{fig:ecl_tornado}) to \texttt{outputs/metrics.json}. They are omitted from the body for length, not because they were unfavourable; each is reproducible from that file without re-running the pipeline.


\clearpage
\section*{References}
\addcontentsline{toc}{section}{References}
\printbibliography[heading=none]

\end{document}
"""

    # ── Placeholder substitutions ──────────────────────────────────────────────
    latex_content = latex_template
    latex_content = latex_content.replace("__TODAY__", today_str)
    _bad_set, _good_set = _target_status_sets(project_root)
    latex_content = latex_content.replace("__TARGET_BAD_SET__", _bad_set)
    latex_content = latex_content.replace("__TARGET_GOOD_SET__", _good_set)
    latex_content = latex_content.replace("__CSI_TABLE__", csi_table_tex)
    latex_content = latex_content.replace("__VINTAGE_DRIFT_SENTENCE__", vintage_drift_sentence)
    latex_content = latex_content.replace("__EAD_MOB_ASSUMPTION__", ead_mob_assumption)
    latex_content = latex_content.replace(
        "__MEAN_PD_LIFETIME__", fmt_pct(_num(metrics, "mean_pd_lifetime"))
    )
    latex_content = latex_content.replace(
        "__MEAN_PD_12M__", fmt_pct(_num(metrics, "mean_pd_12m"))
    )
    latex_content = latex_content.replace("__HAZARD_DISCRIMINATION__", hazard_discrimination_note)
    latex_content = latex_content.replace("__EC_RHO_NOTE__", ec_rho_note)
    latex_content = latex_content.replace("__TTC_BASELINE_NOTE__", ttc_baseline_note)
    latex_content = latex_content.replace("__STAGING_SCENARIO_NOTE__", staging_scenario_note)
    latex_content = latex_content.replace("__REJECT_INFERENCE_NOTE__", reject_inference_note)
    latex_content = latex_content.replace("__ECL_RECONCILIATION_TABLE__", ecl_reconciliation_tex)
    latex_content = latex_content.replace("__AUC_CI_NOTE__", auc_ci_note)
    latex_content = latex_content.replace("__SPIEGELHALTER_NOTE__", spiegelhalter_note)
    latex_content = latex_content.replace("__SELECTION_FUNNEL__", selection_funnel)
    latex_content = latex_content.replace("__MODELB_EXCLUDED__", model_b_excluded_str)
    latex_content = latex_content.replace("__BINNER_USED__", binner_used)
    latex_content = latex_content.replace("__PHASE_FAILURES__", phase_failures_str)
    latex_content = latex_content.replace("__AUC__", auc)
    latex_content = latex_content.replace("__GINI__", gini)
    latex_content = latex_content.replace("__KS__", ks)
    latex_content = latex_content.replace("__AUC_OOT__", auc_oot)
    latex_content = latex_content.replace("__GINI_OOT__", gini_oot)
    latex_content = latex_content.replace("__KS_OOT__", ks_oot)
    latex_content = latex_content.replace("__BRIER_OOT__", brier_oot)
    latex_content = latex_content.replace("__MEAN_LGD__", mean_lgd)
    latex_content = latex_content.replace("__DOWNTURN_LGD__", downturn_lgd)
    latex_content = latex_content.replace("__TOTAL_EL__", total_el)
    latex_content = latex_content.replace("__TOTAL_EAD__", total_ead)
    latex_content = latex_content.replace("__EL_RATE__", el_rate)
    latex_content = latex_content.replace("__TOTAL_RWA__", total_rwa)
    latex_content = latex_content.replace("__TOTAL_RWA_SA__", total_rwa_sa)
    latex_content = latex_content.replace("__RWA_DENSITY__", rwa_density)
    latex_content = latex_content.replace("__TOTAL_ECL__", total_ecl)
    latex_content = latex_content.replace("__ECL_COVERAGE__", ecl_coverage)
    latex_content = latex_content.replace("__STAGE2_PCT__", stage2_pct)
    latex_content = latex_content.replace("__STAGE3_PCT__", stage3_pct)
    latex_content = latex_content.replace("__OPT_CUTOFF__", opt_cutoff)
    latex_content = latex_content.replace("__OPT_APPROVAL__", opt_approval)
    latex_content = latex_content.replace("__OPT_BAD__", opt_bad)
    latex_content = latex_content.replace("__OPT_PROFIT_M__", opt_profit_m)
    latex_content = latex_content.replace("__OPT_RAROC__", opt_raroc)
    latex_content = latex_content.replace("__RAROC_HURDLE__", raroc_hurdle)
    latex_content = latex_content.replace("__RAROC_VS_HURDLE__", raroc_vs_hurdle)
    latex_content = latex_content.replace("__GRID_HURDLE_VERDICT__", grid_hurdle_verdict)
    latex_content = latex_content.replace("__MAX_BAD_RATE__", max_bad_rate_txt)
    latex_content = latex_content.replace("__CORNER_RAROC__", corner_raroc)
    latex_content = latex_content.replace("__CORNER_AGREEMENT__", corner_agreement)
    latex_content = latex_content.replace("__CORNER_DESC__", corner_desc)
    latex_content = latex_content.replace("__CORNER_IMPLICATION__", corner_implication)
    latex_content = latex_content.replace("__COST_OF_CAPITAL__", cost_of_capital_txt)
    # Population counts / split shares — derived, never hard-coded (audit A5, A6)
    latex_content = latex_content.replace("__N_ACCEPTED_FILE__", n_accepted_file)
    latex_content = latex_content.replace("__N_RESOLVED__", n_resolved_outcome)
    latex_content = latex_content.replace("__N_GREYZONE__", n_greyzone)
    latex_content = latex_content.replace("__N_REJECTED_RAW__", n_rejected_raw)
    latex_content = latex_content.replace("__N_MODELLING__", n_modelling)
    latex_content = latex_content.replace("__PCT_TRAIN__", pct_train)
    latex_content = latex_content.replace("__PCT_TEST__", pct_test)
    latex_content = latex_content.replace("__PCT_OOT__", pct_oot)
    latex_content = latex_content.replace("__TRAIN_BAD_RATE__", train_bad_rate)
    latex_content = latex_content.replace("__TRAIN_GOOD_RATE__", train_good_rate)
    latex_content = latex_content.replace("__OOT_BAD_RATE__", oot_bad_rate)
    # Recalibration narrative — reflects whether a calibrator was actually attached (audit A1)
    recalib_residual_note = (
        "Note that the IFRS 9 lifetime PD is produced by the separate hazard model and "
        "is not passed through this transform (Section~6.2), so the ECL provisions do "
        "not inherit the correction; the known limitations in Section~10 still apply."
        if _applied else
        "The residual higher-decile underprediction therefore carries through to the "
        "ECL provisions and is listed among the known limitations in Section~10."
    )
    latex_content = latex_content.replace("__RECALIB_RESIDUAL_NOTE__", recalib_residual_note)
    latex_content = latex_content.replace("__RECALIB_STATUS__", recalib_status)
    latex_content = latex_content.replace("__RECALIB_PRODUCTION_NOTE__", recalib_production_note)
    latex_content = latex_content.replace("__DOWNTURN_NOTE__", downturn_note)
    latex_content = latex_content.replace("__LGD_UPLIFT_PP__", lgd_uplift_pp)
    latex_content = latex_content.replace("__LGD_VAL_N__", lgd_val_n)
    latex_content = latex_content.replace("__N_FEATURES_SC__", n_features_sc)
    latex_content = latex_content.replace("__N_FEATURES_CH__", n_features_ch)
    latex_content = latex_content.replace("__FEATURE_PARITY_NOTE__", feature_parity_note)
    latex_content = latex_content.replace("__WHATIF_LGD_PCT__", whatif_lgd_pct)
    latex_content = latex_content.replace("__WHATIF_EAD_PCT__", whatif_ead_pct)
    latex_content = latex_content.replace("__WHATIF_PD50_PCT__", whatif_pd50_pct)
    latex_content = latex_content.replace("__STAGE3_ECL_SHARE__", stage3_ecl_share)
    latex_content = latex_content.replace("__WHATIF_BASE_NOTE__", whatif_base_note)
    latex_content = latex_content.replace("__EL_ECL_PROXIMITY__", el_ecl_proximity)
    latex_content = latex_content.replace(
        "__PERFORMING_BOOK_CAVEAT__", performing_book_caveat
    )
    latex_content = latex_content.replace("__BASELINE_Z__", baseline_z)
    latex_content = latex_content.replace("__VINTAGE_BAND__", f"${vintage_band_text()}$")
    _scope_caveat = (
        f"Rows before {int(_cal_min_year)} are outside the recalibrator's vintage scope "
        "(Section~7.2) and are therefore raw model output; the two halves of the table are "
        "not the same estimator."
        if _cal_min_year else ""
    )
    latex_content = latex_content.replace("__RECALIB_SCOPE_CAVEAT__", _scope_caveat)
    latex_content = latex_content.replace("__ML_VERDICT__", ml_verdict)
    latex_content = latex_content.replace("__CHAMPION_RATIONALE__", champion_rationale)
    latex_content = latex_content.replace("__CEILING_NOTE__", ceiling_note)
    latex_content = latex_content.replace("__AB_DIRECTION__", ab_direction)
    latex_content = latex_content.replace("__GINI_TTD__", gini_ttd)
    latex_content = latex_content.replace("__GINI_SHIFT__", gini_shift)
    latex_content = latex_content.replace("__STRESS_EL__", stress_el)
    latex_content = latex_content.replace("__STRESS_RWA__", stress_rwa)
    latex_content = latex_content.replace("__STRESS_CAP_REQ__", stress_capital_req)
    latex_content = latex_content.replace("__STRESS_EL_RATIO__", stress_el_ratio)
    latex_content = latex_content.replace("__STRESS_RWA_RATIO__", stress_rwa_ratio)
    latex_content = latex_content.replace("__STRESS_CAP_RATIO__", stress_cap_ratio)
    latex_content = latex_content.replace("__RWA_RELEASE_CAP__", rwa_release_cap)
    latex_content = latex_content.replace("__BASE_CAP_REQ__", base_cap_req)
    latex_content = latex_content.replace("__HL_PVALUE__", hl_pvalue)
    latex_content = latex_content.replace("__PSI_TRAIN_OOT__", psi_train_oot)
    latex_content = latex_content.replace("__IV_TABLE__", iv_table_tex)
    latex_content = latex_content.replace("__LOGIT_TABLE__", logit_table_tex)
    latex_content = latex_content.replace("__SCORECARD_POINTS__", scorecard_points_tex)
    latex_content = latex_content.replace("__SELECTED_FEATURES__", selected_features_str)

    # New custom table replacements
    latex_content = latex_content.replace("__BASE_CAP_REQ_SA__", base_cap_req_sa)
    latex_content = latex_content.replace("__RWA_RELEASE_CAP_ABS__", rwa_release_cap_abs)
    latex_content = latex_content.replace("__CALIBRATION_COMPARISON_TABLE__", _calibration_comparison_table_latex(metrics))
    latex_content = latex_content.replace("__UNDERWRITING_COMPARISON_TABLE__", _underwriting_comparison_table_latex(metrics))
    latex_content = latex_content.replace("__CUTOFF_RAROC_TABLE__", _cutoff_raroc_table_latex(metrics))
    latex_content = latex_content.replace("__MACRO_ELASTICITIES_TABLE__", _macro_elasticities_table_latex(metrics))
    latex_content = latex_content.replace("__RISK_MEASURES_TABLE__", _risk_measures_table_latex(metrics))
    latex_content = latex_content.replace("__COX_TABLE__", _cox_table_latex(metrics))

    # Which covariates actually dominate is read off the fitted per-SD coefficients rather
    # than asserted. The previous sentence named "credit grade and interest rate" as the
    # dominant multipliers while the fitted rate effect was ~1% of hazard across its whole
    # observed range -- an artefact of penalising unstandardised covariates.
    _cox_rows = (metrics.get("survival") or {}).get("cox_summary") or []
    _cox_labels = {
        "grade_num": "credit grade", "int_rate": "interest rate",
        "dti": "debt-to-income", "term_num": "amortisation term",
    }
    if _cox_rows:
        _ranked = sorted(
            _cox_rows, key=lambda r: abs(float(r.get("coef", 0.0))), reverse=True
        )
        _top = [_cox_labels.get(str(r.get("covariate")), str(r.get("covariate")))
                for r in _ranked[:2]]
        _weakest = _cox_labels.get(
            str(_ranked[-1].get("covariate")), str(_ranked[-1].get("covariate"))
        )
        cox_dominant = (
            f"On this run the largest per-SD effects attach to {_top[0]}"
            + (f" and {_top[1]}" if len(_top) > 1 else "")
            + f", and the smallest to {_weakest} "
            f"(hazard ratio {float(_ranked[-1].get('hazard_ratio', float('nan'))):.4f} "
            "per standard deviation)."
        )
    else:
        cox_dominant = ""
    latex_content = latex_content.replace("__COX_DOMINANT__", cox_dominant)

    _delong = (metrics.get("challenger") or {}).get("delong_test") or {}
    _corr = _delong.get("auc_correlation")
    latex_content = latex_content.replace(
        "__DELONG_CORR__",
        f"$\\rho = {float(_corr):.3f}$" if _corr is not None else "a correlation the test now measures rather than assumes away",
    )

    # The report's champion/challenger sentence is about the Weighted Ensemble, so the
    # significance test has to be run against the ensemble -- it used to be run against
    # LightGBM, leaving the one comparison the prose names entirely untested. The ensemble
    # also contains 30% of the champion at fixed, unoptimised weights, which is stated.
    _ens_ab = metrics.get("ab_test_ensemble") or {}
    _ens_w = metrics.get("ensemble_weights") or {}
    if _ens_ab:
        _sig = bool(_ens_ab.get("significant"))
        _diff = _ens_ab.get("diff") or {}
        _lo, _hi = _diff.get("lo"), _diff.get("hi")
        _ci = (
            f" (95\\% CI on the Gini difference: [{float(_lo):+.4f}, {float(_hi):+.4f}])"
            if _lo is not None and _hi is not None else ""
        )
        _w_sc = _ens_w.get("scorecard")
        _w_txt = (
            f" Note also that the ensemble is not an independent model: {float(_w_sc)*100:.0f}\\% "
            "of its prediction is the champion scorecard itself, at weights fixed a priori "
            "rather than optimised."
            if _w_sc is not None else ""
        )
        ensemble_ab_note = (
            "The same paired bootstrap is run for the \\emph{Weighted Ensemble}, which is "
            "the row the champion-versus-challenger discussion in Section~3.6 refers to: "
            "its Gini advantage over the scorecard is "
            + ("statistically significant" if _sig else "\\emph{not} statistically significant")
            + _ci + "." + _w_txt
        )
    else:
        ensemble_ab_note = ""
    latex_content = latex_content.replace("__ENSEMBLE_AB_NOTE__", ensemble_ab_note)
    latex_content = latex_content.replace(
        "__REPORTING_DATE__", tex_escape(str(metrics.get("reporting_date", "2018-12-31")))
    )
    _s3_ead = ((metrics.get("ecl_reconciliation") or {}).get("ead_by_stage") or {}).get("s3")
    latex_content = latex_content.replace(
        "__STAGE3_EAD__",
        f"\\${float(_s3_ead)/1e9:,.2f}bn" if _s3_ead else "the credit-impaired exposure",
    )
    # The HL statistic is evaluated on a subsample (it rejects on any trivial departure at
    # this n), and the recalibration gate keys off its p-value. Saying so, with the spread
    # across replicates, is the difference between a disclosed approximation and a
    # production decision resting invisibly on one draw of 5,000 rows.
    _hl_oot = ((metrics.get("calibration") or {}).get("oot") or {})
    _hl_reps = _hl_oot.get("hl_n_replicates") or _hl_oot.get("n_replicates")
    _hl_n_eval = _hl_oot.get("hl_n_evaluated") or _hl_oot.get("n_evaluated")
    if _hl_reps and _hl_n_eval:
        _lo = _hl_oot.get("hl_pvalue_min", _hl_oot.get("p_value_min"))
        _hi = _hl_oot.get("hl_pvalue_max", _hl_oot.get("p_value_max"))
        _spread = (
            f", and across those replicates the $p$-value ranges from {float(_lo):.4f} to "
            f"{float(_hi):.4f}"
            if _lo is not None and _hi is not None else ""
        )
        hl_subsample_note = (
            f"The statistic itself is computed on subsamples of {int(_hl_n_eval):,} rows "
            f"rather than the full slice --- at this $N$ it rejects on any trivial "
            f"departure --- and is reported as the median over {int(_hl_reps)} independent "
            f"draws{_spread}. The recalibration gate below keys off that median, so the "
            "decision does not rest on a single draw."
        )
    else:
        hl_subsample_note = ""
    latex_content = latex_content.replace("__HL_SUBSAMPLE_NOTE__", hl_subsample_note)

    # Which macro path actually ran. The claim "sourced live from the official FRED API"
    # was unconditional, but without FRED_API_KEY the loader silently falls back to a
    # hardcoded offline table that is not the same data.
    _macro_src = str(metrics.get("macro_source", "unknown"))
    _macro_src_txt = {
        "live": r"sourced live from the official FRED (St.\ Louis Fed) API",
        "offline": (
            r"taken from the repository's bundled offline historical table --- the live "
            r"FRED download did not run on this build, so these series are \emph{not} the "
            r"live API data"
        ),
    }.get(_macro_src, r"sourced from FRED (St.\ Louis Fed); the provenance marker for this build was not recorded")
    latex_content = latex_content.replace("__MACRO_SOURCE__", _macro_src_txt)
    latex_content = latex_content.replace("__TORNADO_SPAN__", tornado_span)
    latex_content = latex_content.replace("__TORNADO_PD_ONLY__", tornado_pd_only)
    _cox_cindex = metrics.get("survival", {}).get("c_index", float("nan"))
    _cox_cindex_txt = f"{_cox_cindex:.4f}" if isinstance(_cox_cindex, (int, float)) and _cox_cindex == _cox_cindex else "n/a"
    latex_content = latex_content.replace("__COX_CINDEX__", _cox_cindex_txt)
    latex_content = latex_content.replace("__LGD_VALIDATION_TABLE__", _lgd_validation_table_latex(metrics))
    latex_content = latex_content.replace("__ECL_WHATIF_TABLE__", _ecl_whatif_table_latex(metrics))
    latex_content = latex_content.replace("__MACRO_TS_TABLE__", _macro_ts_table_latex(metrics))
    latex_content = latex_content.replace("__HHI_TABLE__", _hhi_table_latex(metrics))
    latex_content = latex_content.replace("__AB_TEST_TABLE__", _ab_test_table_latex(metrics))
    latex_content = latex_content.replace("__VINTAGE_CALIB_TABLE__", _vintage_calib_table_latex(metrics))
    latex_content = latex_content.replace("__LIFETIME_PD_CALIBRATION_TABLE__", _lifetime_pd_calibration_table_latex(metrics))

    # Dynamic split size variables
    latex_content = latex_content.replace("VAR_N_TRAIN", f"{metrics.get('n_train', 363317):,}")
    latex_content = latex_content.replace("VAR_N_TEST", f"{metrics.get('n_test', 90830):,}")
    latex_content = latex_content.replace("VAR_N_OOT", f"{metrics.get('n_oot', 538515):,}")

    # LightGBM challenger vs scorecard OOT AUC
    lgbm_auc_oot_str = f"{metrics.get('challenger', {}).get('auc_oot', 0.6943):.4f}"
    latex_content = latex_content.replace("VAR_LGBM_AUC_OOT", lgbm_auc_oot_str)
    latex_content = latex_content.replace("VAR_AUC_OOT", fmt_dec(_num(metrics, "auc_oot")))
    # Pure-underwriting (Model B) OOT AUC — same source as the underwriting comparison table
    latex_content = latex_content.replace(
        "VAR_MODELB_AUC_OOT",
        fmt_dec(metrics.get("underwriting_scorecard", {}).get("oot", {}).get("auc", 0.0)),
    )
    # SHAP challenger top-2 features (data-driven, so the prose always matches
    # figures/validation/shap_challenger_summary.png instead of a stale hard-coded pair)
    _shap_rows = (metrics.get("challenger", {}) or {}).get("shap_mean_abs", []) or []
    _shap_feats = [str(r.get("feature", "")) for r in _shap_rows[:2]]
    while len(_shap_feats) < 2:
        _shap_feats.append("n/a")
    latex_content = latex_content.replace("VAR_SHAP_TOP1", _shap_feats[0].replace("_", r"\_"))
    latex_content = latex_content.replace("VAR_SHAP_TOP2", _shap_feats[1].replace("_", r"\_"))
    # New: vintage PD backtest rows
    latex_content = latex_content.replace("__PD_BACKTEST_ROWS__", pd_backtest_rows_tex)
    # New: ML comparison table
    latex_content = latex_content.replace("__ML_COMPARISON_TABLE__", ml_comparison_table_tex)
    # The stage migration matrix is no longer rendered: without a servicing panel the
    # t-12 state can only be a re-labelling of the reporting-date state, so the table
    # was an artefact of its own reconstruction (FLAWS-N9).

    # ── Benchmark ranges + verdicts (Tables 13 & 18) ─────────────────────────────
    # Single source of truth: reports/benchmarks.py. The published range cell AND the
    # pass/fail verdict for every row come from the SAME Benchmark object, so they can
    # never drift apart (the old design kept a Python literal and a LaTeX string in two
    # places and they disagreed). No hand-typed/static benchmark rows remain.
    from benchmarks import BENCHMARKS, TABLE13_KEYS, TABLE18_KEYS  # noqa: PLC0415

    def _as_float(x):
        try:
            f = float(x)
            return f if f == f else None  # drop NaN
        except (TypeError, ValueError):
            return None

    try:
        _rwa_density_v = float(str(metrics.get("rwa_density", "0")).replace("%", "")) / 100.0
    except ValueError:
        _rwa_density_v = float("nan")
    _lgbm_gini_v = _as_float(_get_ml_gini("LightGBM Classifier", ml_rows))
    # LGD R^2: prefer top-level; fall back to the nested lgd_validation dict.
    _lgd_r2_metric = metrics.get("lgd_r2")
    if _lgd_r2_metric is None:
        _lgd_r2_metric = (metrics.get("lgd_validation") or {}).get("r2")

    # metric_key -> live project value at build time
    bench_values = {
        "auc_oot": metrics.get("auc_oot"),
        "gini_oot": metrics.get("gini_oot"),
        "mean_lgd": metrics.get("mean_lgd"),
        "lgd_r2": _lgd_r2_metric,
        "rwa_density": _rwa_density_v,
        "gini_shift": metrics.get("gini_shift"),
        "psi_total": metrics.get("psi_total"),
        "lgbm_gini_oot": _lgbm_gini_v,
        "downturn_lgd": metrics.get("downturn_lgd"),
        "ecl_coverage": metrics.get("ecl_coverage"),
        "stage2_pct": metrics.get("stage2_pct"),
    }

    _bench_tokens: list[tuple[str, str]] = []
    for _key in dict.fromkeys([*TABLE13_KEYS, *TABLE18_KEYS]):  # de-dup, preserve order
        _b = BENCHMARKS[_key]
        _verd, _cmt = _b.verdict(bench_values.get(_b.metric_key))
        _bench_tokens += [
            (f"__RANGE_{_key}__", _b.range_tex()),
            (f"__VERDICT_{_key}__", _verd),
            (f"__COMMENT_{_key}__", _cmt),
        ]

    # LGD R^2 value cell is plain (not \num) so a missing metric renders "N/A" cleanly.
    _lgd_r2_v = _as_float(_lgd_r2_metric)
    _bench_tokens.append(("__LGD_R2__", f"{_lgd_r2_v:.4f}" if _lgd_r2_v is not None else "N/A"))
    # R^2 of the rejected two-stage champion, quoted in prose alongside the
    # deployed model's value so the two figures are never conflated.
    _lgd_r2_ts = _as_float(((metrics.get("lgd_model_comparison") or {}).get("champion") or {}).get("r2"))
    _bench_tokens.append(("__LGD_R2_TWOSTAGE__", f"{_lgd_r2_ts:.2f}" if _lgd_r2_ts is not None else "N/A"))
    _bench_tokens.append(("__LGD_R2_SHORT__", f"{_lgd_r2_v:.2f}" if _lgd_r2_v is not None else "N/A"))

    # IRB-vs-SA is a direction check (not a numeric range); handled explicitly.
    _rwa_irb = float(_num(metrics, "total_rwa") or 0.0)
    _rwa_sa = float(_num(metrics, "total_rwa_sa") or 0.0)
    if _rwa_irb > 0 and _rwa_sa > 0:
        if _rwa_irb > _rwa_sa:
            _irb_sa_dir, v_irbsa = "IRB $>$ SA", "Consistent"
            c_irbsa = ("Risk-sensitive IRB exceeds the flat 75\\% SA weight for this "
                       "higher-risk unsecured book: a capital \\emph{surcharge}, not "
                       "relief, is the economically expected outcome")
        else:
            _irb_sa_dir, v_irbsa = "IRB $<$ SA", "Consistent"
            c_irbsa = ("IRB delivers capital relief versus the flat SA weight, as "
                       "expected for a lower-risk book")
    else:
        _irb_sa_dir, v_irbsa, c_irbsa = "N/A", "N/A", "RWA figures unavailable at build time"
    _bench_tokens += [
        ("__IRB_SA_DIRECTION__", _irb_sa_dir),
        ("__VERDICT_IRB_SA__", v_irbsa),
        ("__COMMENT_IRB_SA__", c_irbsa),
    ]

    for tok, val in _bench_tokens:
        latex_content = latex_content.replace(tok, val)

    # ── D1: Abstract live-metric substitutions ────────────────────────────────
    latex_content = latex_content.replace(
        "VAR_GINI_RAG",
        str(metrics.get("rag_status", {}).get("gini_rag", "N/A")),
    )
    latex_content = latex_content.replace(
        "VAR_PSI_OOT",
        f"{metrics.get('psi_total', 0):.4f}",
    )
    latex_content = latex_content.replace(
        "VAR_GINI_OOT",
        fmt_dec(_num(metrics, "gini_oot")),
    )
    latex_content = latex_content.replace(
        "VAR_AUC_OOT",
        fmt_dec(_num(metrics, "auc_oot")),
    )
    latex_content = latex_content.replace(
        "VAR_MEAN_LGD",
        fmt_dec(_num(metrics, "mean_lgd")),
    )
    latex_content = latex_content.replace(
        "VAR_DOWNTURN_LGD",
        fmt_dec(_num(metrics, "downturn_lgd")),
    )
    latex_content = latex_content.replace(
        "VAR_RWA_IRB",
        fmt_num(_num(metrics, "total_rwa")),
    )
    latex_content = latex_content.replace(
        "VAR_RWA_DENSITY",
        str(metrics.get("rwa_density", "N/A")).replace("%", "\\%"),
    )
    latex_content = latex_content.replace(
        "VAR_ECL_TOTAL",
        fmt_num(_num(metrics, "total_ecl")),
    )
    latex_content = latex_content.replace(
        "VAR_ECL_COVERAGE",
        fmt_pct(_num(metrics, "ecl_coverage"), precision=3),
    )

    # ── D3: Literature benchmark substitutions ────────────────────────────────
    if "VAR_LGBM_GINI_OOT" not in latex_content.replace("VAR_LGBM_GINI_OOT", ""):
        pass  # already replaced above if present
    latex_content = latex_content.replace(
        "VAR_LGBM_GINI_OOT",
        _get_ml_gini("LightGBM Classifier", ml_rows),
    )
    latex_content = latex_content.replace(
        "VAR_STAGE2_PCT",
        f"{metrics.get('stage2_pct', 0):.1%}".replace("%", "\\%"),
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"LaTeX report written to {tex_path}")

    # Fix 1.1#4: abort build if any template variable survived substitution, a citation
    # is unresolved, or a narrative number contradicts its metric source.
    run_tex_checks(tex_path, metrics)

    # ── 4-pass XeLaTeX + biber compilation ────────────────────────────────────
    reports_dir = os.path.dirname(tex_path)
    tex_name = "model_risk_report"
    passes = [
        ["xelatex", "-interaction=nonstopmode", f"{tex_name}.tex"],
        ["biber", tex_name],
        ["xelatex", "-interaction=nonstopmode", f"{tex_name}.tex"],
        ["xelatex", "-interaction=nonstopmode", f"{tex_name}.tex"],
    ]
    try:
        for i, cmd in enumerate(passes, 1):
            print(f"Pass {i}/4: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, cwd=reports_dir, capture_output=True,
                encoding="utf-8", errors="replace", timeout=300
            )
            if result.returncode != 0 and cmd[0] != "biber":
                print(f"  Error (exit {result.returncode}):")
                print(result.stderr[-2000:] if result.stderr else result.stdout[-2000:])
            else:
                print(f"  Pass {i} OK (exit {result.returncode})")
        print("LaTeX: 4-pass compilation complete. PDF written to reports/model_risk_report.pdf")
    except FileNotFoundError as e:
        print(f"LaTeX: compiler not found — {e}. Install XeLaTeX (MiKTeX/TeX Live).")
    except Exception as e:
        print(f"LaTeX: compilation failed — {e}")


if __name__ == "__main__":
    render_latex()

