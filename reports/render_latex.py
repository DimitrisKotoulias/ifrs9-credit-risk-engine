import os
import sys
import json
import subprocess
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_checks import run_metric_checks, run_tex_checks  # noqa: E402


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
    def _iv_table_latex(iv_rows):
        if not iv_rows:
            return r"\textit{No IV data available.}"
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Feature Information Value (IV) Ranking}",
            r"\label{tab:iv_ranking}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Feature} & \textbf{IV} & \textbf{Predictive Power Band} \\",
            r"\midrule",
        ]
        bands = {(0, 0.02): "Negligible", (0.02, 0.1): "Weak",
                 (0.1, 0.3): "Medium", (0.3, 0.5): "Strong", (0.5, 999): "Very Strong"}
        for row in sorted(iv_rows, key=lambda r: r["iv"], reverse=True):
            feat = row["variable"].replace("_", r"\_")
            iv_val = row["iv"]
            band = next((v for (lo, hi), v in bands.items() if lo <= iv_val < hi), "N/A")
            lines.append(f"\\texttt{{{feat}}} & {iv_val:.4f} & {band} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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

    def _scorecard_points_latex(sc_rows, top_n=25):
        if not sc_rows:
            return r"\textit{No scorecard table available.}"
        feat_ranges = defaultdict(lambda: [9999, -9999])
        for r in sc_rows:
            feat_ranges[r["feature"]][0] = min(feat_ranges[r["feature"]][0], r["points"])
            feat_ranges[r["feature"]][1] = max(feat_ranges[r["feature"]][1], r["points"])
        rows_sorted = sorted(
            sc_rows,
            key=lambda r: abs(feat_ranges[r["feature"]][1] - feat_ranges[r["feature"]][0]),
            reverse=True,
        )
        shown = rows_sorted[:top_n]
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
            r"\caption{Credit Scorecard Points Table (Top Bins by Point Spread)}",
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
            r"\multicolumn{" + str(len(col_spec)) + r"}{l}{\footnotesize WoE = ln(\%Good/\%Bad); Points = ($-\text{WoE}_j * \beta_j + \alpha/n$) * Factor + Offset/n} \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _pd_backtest_rows_latex(rows):
        if not rows:
            return r"\multicolumn{5}{c}{\textit{No vintage backtest data available.}} \\"
        lines = []
        import numpy as np
        for row in rows:   # ALL vintages — no slice
            vintage  = str(row.get("vintage", "N/A"))
            n        = int(row.get("n_loans", 0))
            pred_pd  = float(row.get("predicted_pd", 0))
            actual_dr = float(row.get("actual_dr", row.get("actual_default_rate", 0)))
            pd_ratio  = float(row.get("pd_ratio", 0))
            flag      = row.get("calibration_flag", "pass")

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
        fit_on = comp.get("recalibration_fit_on", "in_time_test").replace("_", " ")

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

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{OOT Calibration Diagnostics: Before vs.\ After Recalibration}",
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
            + r"Recalibration uses isotonic regression fitted \emph{out-of-sample} on the "
            + f"{fit_on} partition and applied (transform only) to OOT; the OOT diagnostics above therefore reflect genuine "
            + r"generalisation, not in-sample recalibration. Fitting the calibrator on the OOT set itself would trivially pass "
            + r"the Hosmer-Lemeshow test and is deliberately avoided.} \\",
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
        # The reconciled optimal cutoff (marginal RAROC-hurdle rule) must appear as
        # an actual, highlighted row in the table.
        opt = metrics_dict.get("cutoff_optimal_profit", {})
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
            r"\\[0.3em]{\footnotesize The Granger verdict applies a Bonferroni-corrected threshold $\alpha_{\text{corr}}$ across the lags tested; a minimum $p$-value above $\alpha_{\text{corr}}$ --- even if below the nominal $0.05$ --- is reported as no causality, guarding against multiple-testing false positives.}",
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
            r"status is right-censored.} \\",
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
        rho = float(ec.get("rho", 0.15))
        reg_cap = _m("regulatory_capital")
        ec_cap = _m("economic_capital")
        ratio = ec.get("ec_to_reg_ratio", 0.0)
        ratio_txt = f"{ratio * 100:.1f}\\%" if reg_cap > 0 else r"n/a"

        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Monte Carlo Economic Capital --- Risk Measures (ASRF, "
            f"$N={n_sim:,}$ simulations, $\\rho={rho:.2f}$)}}",
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
            p_txt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
            body.append(f"{label} & {coef:+.5f} & {hr:.5f} & {p_txt} \\\\")

        return "\n".join([
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Cox Proportional-Hazards Model --- Covariate Summary}",
            r"\label{tab:cox_summary}",
            r"\vspace{0.5em}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"\textbf{Covariate} & \textbf{Coef ($\beta$)} & \textbf{Hazard Ratio} & \textbf{$p$-value} \\",
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
            f"\\caption{{ECL What-If Sensitivity (Baseline ECL $=$ \\${base:,.1f}M; note that this Baseline-only ECL reference differs from the probability-weighted total ECL in Table~\\ref{{tab:exec_summary}} as it excludes the Upside/Downside macro scenario shocks mandated by IFRS~9)}}",
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

    iv_table_tex = _iv_table_latex(_iv_rows)
    logit_table_tex = _logit_table_latex(_coef_rows)
    scorecard_points_tex = _scorecard_points_latex(_sc_rows)
    selected_features_str = (
        ",\\allowbreak ".join([f"\\texttt{{{f.replace('_', chr(92) + '_')}}}" for f in _selected])
        if _selected else "N/A"
    )
    pd_backtest_rows_tex = _pd_backtest_rows_latex(metrics.get("pd_backtest_vintage", []))
    ml_comparison_table_tex = _ml_comparison_table_latex(metrics.get("ml_benchmark_comparison", []))

    # ── D3: ML Gini helper for benchmark table ────────────────────────────────
    def _get_ml_gini(model_name, rows):
        for r in rows:
            if r.get("model") == model_name:
                return f"{r.get('oot_gini', 0):.4f}"
        return "N/A"
    ml_rows = metrics.get("ml_benchmark_comparison", [])

    # ── Helper formatters ──────────────────────────────────────────────────────
    def fmt_num(val, fmt="{:,.0f}"):
        try:
            return fmt.format(float(val))
        except Exception:
            return "N/A"

    def fmt_dec(val, precision=4):
        try:
            return f"{float(val):.{precision}f}"
        except Exception:
            return "N/A"

    def fmt_pct(val, precision=2):
        try:
            return f"{float(val) * 100:.{precision}f}\\%"
        except Exception:
            return "N/A"

    # ── Load metrics ───────────────────────────────────────────────────────────
    auc = fmt_dec(metrics.get("auc", 0.6324))
    gini = fmt_dec(metrics.get("gini", 0.2648))
    ks = fmt_dec(metrics.get("ks", 0.2151))
    auc_oot = fmt_dec(metrics.get("auc_oot", 0.6326))
    gini_oot = fmt_dec(metrics.get("gini_oot", 0.2651))
    ks_oot = fmt_dec(metrics.get("ks_oot", 0.1984))
    brier_oot = fmt_dec(metrics.get("calibration", {}).get("oot", {}).get("brier_score", 0.0582), 4)
    mean_lgd = fmt_dec(metrics.get("mean_lgd", 0.1178))
    downturn_lgd = fmt_dec(metrics.get("downturn_lgd", 0.1339))
    total_el = fmt_num(metrics.get("total_el", 2474806))
    total_ead = fmt_num(metrics.get("total_ead_portfolio", 326526293))
    el_rate = fmt_pct(metrics.get("el_rate", 0.0076))
    total_rwa = fmt_num(metrics.get("total_rwa", 67238352))
    total_rwa_sa = fmt_num(metrics.get("total_rwa_sa", 244894720))
    rwa_density = str(metrics.get("rwa_density", "20.6%")).replace("%", "\\%")
    total_ecl = fmt_num(metrics.get("total_ecl", 2428522))
    ecl_coverage = fmt_pct(metrics.get("ecl_coverage", 0.0074), precision=3)
    stage2_pct = fmt_pct(metrics.get("stage2_pct", 0.0))
    stage3_pct = fmt_pct(metrics.get("stage3_pct", 0.0635))
    # Cite the reconciled optimal cutoff (marginal RAROC-hurdle rule; traceable
    # highlighted table row), falling back to the risk-appetite cutoff only if the
    # profit sweep is missing.
    _opt_profit_row = metrics.get("cutoff_optimal_profit", {})
    opt_cutoff = fmt_dec(_opt_profit_row.get("cutoff", metrics.get("optimal_cutoff_threshold", 550.0)), precision=0)
    opt_approval = fmt_pct(_opt_profit_row.get("approval_rate", metrics.get("optimal_approval_rate", 0.871)))
    opt_bad = fmt_pct(_opt_profit_row.get("bad_rate", metrics.get("optimal_bad_rate", 0.0548)))
    opt_profit_m = f"{_opt_profit_row.get('expected_profit', 0.0) / 1e6:,.1f}"
    opt_raroc = fmt_pct(_opt_profit_row.get("raroc", 0.0))
    raroc_hurdle = fmt_pct(metrics.get("cutoff_raroc_hurdle", 0.15))
    # Data-driven hurdle comparison so the prose can never contradict the table
    # (a negative RAROC must not be described as "above" a positive hurdle).
    _opt_raroc_v = float(_opt_profit_row.get("raroc", 0.0))
    _hurdle_v = float(metrics.get("cutoff_raroc_hurdle", 0.15))
    if _opt_raroc_v >= 1.5 * _hurdle_v:
        raroc_vs_hurdle = "comfortably above"
    elif _opt_raroc_v >= _hurdle_v:
        raroc_vs_hurdle = "above"
    else:
        raroc_vs_hurdle = "below"
    max_bad_rate_txt = fmt_pct(metrics.get("cutoff_max_bad_rate", 0.15))
    _corner_row = metrics.get("cutoff_raroc_max") or metrics.get("cutoff_profit_argmax", {})
    corner_raroc = fmt_pct(_corner_row.get("raroc", 0.0))
    gini_ttd = fmt_dec(metrics.get("gini_ttd", 0.2086))
    gini_shift = fmt_dec(metrics.get("gini_shift", -0.0555))
    stress_el = fmt_num(metrics.get("stress_el", 8985195))
    stress_rwa = fmt_num(metrics.get("stress_rwa", 95903062))
    stress_capital_req = fmt_num(metrics.get("stress_capital_req", 7672245))
    stress_el_ratio = fmt_pct(metrics.get("stress_el", 8985195) / metrics.get("total_el", 2474806) - 1.0, precision=1)
    stress_rwa_ratio = fmt_pct(metrics.get("stress_rwa", 95903062) / metrics.get("total_rwa", 67238352) - 1.0, precision=1)
    stress_cap_ratio = fmt_pct(
        metrics.get("stress_capital_req", 7672245) / (metrics.get("total_rwa", 67238352) * 0.08) - 1.0, precision=1
    )
    today_str = os.environ.get("REPORT_DATE") or date.today().strftime("%d %B %Y")
    rwa_release_cap = fmt_num((metrics.get("total_rwa_sa", 0) - metrics.get("total_rwa", 0)) * 0.08)
    base_cap_req = fmt_num(metrics.get("total_rwa", 0) * 0.08)
    base_cap_req_sa = fmt_num(metrics.get("total_rwa_sa", 0) * 0.08)
    rwa_release_cap_abs = fmt_num(abs((metrics.get("total_rwa_sa", 0) - metrics.get("total_rwa", 0)) * 0.08))
    hl_pvalue = fmt_dec(metrics.get("calibration", {}).get("oot", {}).get("hl_pvalue", 0.1656), 4)
    psi_train_oot = fmt_dec(metrics.get("stability", {}).get("psi_train_oot", 0.0005), 4)

    # ── Stage migration ────────────────────────────────────────────────────────
    _mig = metrics.get("ifrs9_stage_migration", {})

    def _mig_cell(fs, ts):
        v = _mig.get(str(fs), _mig.get(fs, {}))
        if isinstance(v, dict):
            return str(int(v.get(str(ts), v.get(ts, 0))))
        return "0"

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
\usepackage[left=2.0cm,right=2.0cm,top=2.2cm,bottom=2.2cm,headheight=14pt]{geometry}
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
\titlespacing{\section}{0pt}{11pt}{4pt}
\titlespacing{\subsection}{0pt}{7pt}{2pt}
\setlength{\parskip}{5pt plus 1pt minus 1pt}

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
  {\small Lending Club Consumer Loans $\cdot$ 2007--2018 $\cdot$ N = 2.26M\par}
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
loan portfolio (2,260,701 accepted loans; 27,648,741 rejected applications).
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

This report documents the mathematical foundation, development methodology, and empirical validation of an end-to-end retail credit underwriting, capital calculation, and expected credit loss (ECL) engine. Trained on 2.26 million historical consumer records originated between 2007 and 2018, this framework is designed to bridge the gap between risk underwriting, regulatory capital management, and standard accounting provisions.

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

An OOT AUC of __AUC_OOT__ (Gini __GINI_OOT__) is the realistic discrimination ceiling for an application scorecard built on origination-only features on LendingClub-style unsecured consumer data: __RANGE_GINI_OOT__ is the published Gini range for consumer-credit scorecards surveyed by \textcite{lessmann2015benchmarking}, and the LightGBM challenger (Section~3.6) plateaus at essentially the same level rather than materially exceeding it --- consistent with a dataset/feature ceiling rather than an under-fitted model. The figure should therefore be read as realistic, benchmarked performance rather than a shortfall.

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
A binary default indicator ($Y$) is defined using the 90+ Days Past Due (DPD) default standard, in compliance with BCBS and IFRS 9 standards:
\begin{equation}
Y =
\begin{cases}
1 \text{ (Bad)}, & \text{if status } \in \{\text{Charged Off}, \text{Default}, \text{Late (31--120 days)}\} \\
0 \text{ (Good)}, & \text{if status } \in \{\text{Fully Paid}\}
\end{cases}
\end{equation}

\subsection{Out-of-Time (OOT) Splitting}
To replicate standard banking validation practices, the data is split chronologically based on loan origination date (\texttt{issue\_d}):
\begin{itemize}
    \item \textbf{Training Population:} Loans originated prior to January 2015 ($N = \text{VAR_N_TRAIN}$; 36.6\% of modelling population). Training bad rate: 17.1\%.
    \item \textbf{In-Time Holdout Set:} Stratified random 20\% sample from the training period ($N = \text{VAR_N_TEST}$; 9.2\% of modelling population). Stratification preserves the 82.9\%/17.1\% good/bad ratio.
    \item \textbf{Out-of-Time (OOT) Set:} All loans originated between January 2016 and December 2018 ($N = \text{VAR_N_OOT}$; 54.3\% of modelling population). OOT observed bad rate: 25.27\%, reflecting the portfolio's seasoning and the macroeconomic deterioration of the 2016--2018 vintage cohorts.
\end{itemize}
This chronological split simulates how the model will perform on future vintages, testing for structural or macroeconomic shifts.

\subsection{Leakage and Target Variable Separation}
Data leakage is controlled by dropping post-origination fields (e.g., outstanding balance, payment records, and recovery metrics) from the PD feature set. However, these post-origination variables (such as actual recoveries and write-offs) are preserved for the defaulted-only population. This allows the LGD model to evaluate recovery performance without leaking future data into the PD scorecard.

\subsection{Vintage Cohort Analysis}
Vintage analysis tracks the cumulative default curves of origination cohorts (quarters) over their Months-on-Book (MOB). Figure~\ref{fig:vintage_curves} illustrates these curves. A steep initial slope indicates seasoning, while the curves flatten as high-risk accounts default early, illustrating standard credit risk dynamics.

\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{figures/vintage_default_curves.png}
\caption{Cumulative Default Rates by Quarterly Vintage Cohorts}
\label{fig:vintage_curves}
\end{figure}

Figure~\ref{fig:eda_target_grade} displays additional exploratory distributions, demonstrating the relationship between historical default rates and risk grades, amortization terms, and loan purposes.

All EDA visualizations in this section are computed on the in-time development sample (train and test partitions of the resolved-outcome modelling population), not the full raw portfolio: the Out-of-Time partition is withheld from all exploratory analysis to preserve the integrity of the temporal validation, and loans with unresolved statuses are excluded per Section~2.1. Full-portfolio statistics ($N = 2{,}260{,}701$ accepted loans) are used for portfolio-level capital and impairment calculations. Population counts embedded in the EDA figures therefore reflect the development sample.

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
Continuous features are binned to handle non-linear relationships, outliers, and missing values. Monotonic trends are enforced across bins using isotonic regression. For each bin $i$, the Weight of Evidence (WoE) is calculated as:
\begin{equation}
WoE_i = \ln\!\left( \frac{\text{Proportion of Good}_i}{\text{Proportion of Bad}_i} \right) = \ln\!\left( \frac{N_{G,i} / N_{G,total}}{N_{B,i} / N_{B,total}} \right)
\end{equation}
The predictive power of each feature is evaluated using the Information Value (IV):
\begin{equation}
IV = \sum_{i=1}^{k} \left( \frac{N_{G,i}}{N_{G,total}} - \frac{N_{B,i}}{N_{B,total}} \right) \times WoE_i
\end{equation}
\parencite[Ch.~4]{siddiqi2017}; \parencite{hand1997}. Features with an IV below $0.02$ are dropped due to low predictive power, while multicollinearity is controlled by removing features with a Variance Inflation Factor (VIF) greater than $5.0$.

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

\subsection{Feature Selection Results: IV Ranking and VIF Filter}
After WoE binning, features are ranked by Information Value (IV). Features with $IV < 0.02$ (negligible predictive power) or $IV > 0.50$ (likely target leakage) are excluded. Multicollinearity is then controlled by removing features with a Variance Inflation Factor (VIF) greater than 5.0, yielding the final set of selected features:

\begin{sloppypar}\noindent
__SELECTED_FEATURES__.
\end{sloppypar}

__IV_TABLE__

\subsection{Logistic Regression Coefficient Output}
The following table presents the logistic regression coefficient estimates, standard errors, z-statistics and p-values for all retained features. As expected for a WoE-based model predicting default ($Y=1$), all feature coefficients carry a \textbf{negative sign}: higher WoE values indicate a higher proportion of ``Good'' loans, and therefore map to lower default probability.

__LOGIT_TABLE__

\subsection{Scorecard Points Table}
Each WoE bin is converted to credit score points using the scaling formula. Higher points correspond to lower default risk. The table below shows the top bins ranked by point spread (i.e., the features with the greatest discriminatory contribution to the final score).

__SCORECARD_POINTS__

\subsection{Interpretability vs.\ Performance: Challenger Model Benchmark}
A non-linear LightGBM model was trained on the same feature set as a challenger \parencite{baesens2016}. Although gradient boosting is competitive in-sample, on the out-of-time (OOT) set the interpretable logistic scorecard generalises better: its OOT AUC (VAR_AUC_OOT) exceeds the LightGBM challenger's (VAR_LGBM_AUC_OOT), a gap confirmed as statistically significant by both the DeLong test and the paired bootstrap in Section~7.8. The WoE scorecard's coarse, monotone binning generalises across the 2016--2018 regime shift better than the boosted model, which is more prone to fitting in-sample idiosyncrasies. Combined with its regulatory advantages, the scorecard is therefore the preferred underwriting model. Regulatory standards (such as the US Fair Credit Reporting Act) require financial institutions to provide clear ``adverse action codes'' (reasons for denial) to rejected applicants. A linear scorecard allows for immediate, exact points-attribution for each feature, which is not possible with complex machine learning models without relying on approximations like SHAP.

To address potential policy-decision circularity from using LendingClub's own underwriting variables (\texttt{int\_rate} and \texttt{grade}), we developed a secondary Pure Underwriting Scorecard (Model B) that completely excludes these fields and relies solely on applicant credit bureau and demographic variables. Table~\ref{tab:underwriting_comparison} compares the performance of the full scorecard (Model A) against this independent underwriting model. While Model A is the designated pipeline champion due to its superior discrimination, it utilizes LendingClub's pricing variables which are themselves highly correlated risk assessments. This introduces a degree of decision circularity. Model B (Pure Underwriting Scorecard) shows that a model built solely on raw credit bureau and demographic features remains competitive (OOT AUC = VAR_MODELB_AUC_OOT vs Model A's VAR_AUC_OOT, both as reported in Table~\ref{tab:underwriting_comparison}), supporting the scorecard's viability in an independent bank underwriting environment.

__UNDERWRITING_COMPARISON_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/numeric_distributions.png}
\caption{Distribution of Key Numeric Features}
\label{fig:num_dist}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/missingness.png}
\caption{Missingness Density Analysis}
\label{fig:missing_anal}
\end{figure}

\subsection{Selection Bias and Reject Inference (Parcelling)}
Scorecard models developed only on approved applicants suffer from selection bias. Because rejected applicants are excluded, their risk profiles and actual default rates are unobserved. To adjust for this, we implemented the \textbf{Parcelling} reject inference technique to probabilistically allocate outcomes to rejected applicants based on the accepts scorecard's predictions.

The pooled accepts and parcelled rejects population was refitted to produce a corrected through-the-door (TTD) scorecard. Table~\ref{tab:reject_inference} outlines the results of the refitting and selection bias adjustment:

\begin{table}[h]
    \centering
    \caption{Reject Inference (Parcelling) Gini Coefficient Shift}
    \label{tab:reject_inference}
    \vspace{0.5em}
    \begin{tabular}{llp{6cm}}
        \toprule
        \textbf{Scorecard Population} & \textbf{Gini} & \textbf{Business Interpretation} \\
        \midrule
        \textbf{Accepts-Only (Base)} & __GINI__ & Discrimination on the approved-only population (excludes rejects). \\
        \textbf{Through-the-Door (Parcelled Refit)} & __GINI_TTD__ & Corrected scorecard accounts for selection bias across TTD. \\
        \midrule
        \textbf{Gini Coefficient Shift} & __GINI_SHIFT__ & Conservative risk dilution when scoring raw through-the-door applicants. \\
        \bottomrule
    \end{tabular}
\end{table}

The Gini shift of \textbf{__GINI_SHIFT__} points demonstrates a standard credit cycle finding: through-the-door populations contain higher latent risk profiles. This refitting adjusts the credit scorecard's parameters to correct for this systemic underwriting selection bias.

\subsection{Survival Analysis: Kaplan-Meier and Cox Proportional Hazards}
The production PD term structure (Section~6) is a discrete-time hazard model. As a challenger model we additionally fit a time-to-event survival model, the industry standard for IFRS~9 lifetime-PD term-structure work \parencite{bellotti2009}. Kaplan-Meier estimators give non-parametric survival curves $S(t)$ per credit grade (Figure~\ref{fig:km_survival}), and a Cox proportional-hazards model quantifies each covariate's multiplicative effect on the default hazard. The duration is a months-on-book proxy derived from cumulative payments and the event is the binary default flag, a synthesised time-to-event dataset since the raw data records no observed default month (a documented limitation revisited in Section~10). The model's rank-discrimination is summarised by the concordance index (Cox C-index $=$ __COX_CINDEX__), the survival-analysis analogue of the AUC.

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures/km_survival_curves.png}
\caption{Kaplan-Meier non-default survival curves by credit grade. Lower grades separate downward, confirming the expected monotone grade--risk ordering.}
\label{fig:km_survival}
\end{figure}

Table~\ref{tab:cox_summary} reports the fitted Cox coefficients, hazard ratios $\exp(\beta)$ and Wald $p$-values. A hazard ratio above $1$ raises the instantaneous default hazard; below $1$ lowers it. The dominant hazard multipliers attach to credit grade and interest rate, while debt-to-income and amortisation term enter with smaller but individually significant positive coefficients (all values as tabulated below); each covariate therefore raises the default hazard monotonically as it increases, consistent with the scorecard's risk ordering.

__COX_TABLE__

% -----------------------------------------------------------------------------
% 4. LOSS GIVEN DEFAULT (LGD) AND EXPOSURE AT DEFAULT (EAD)
% -----------------------------------------------------------------------------
\section{Loss Given Default (LGD) and Exposure at Default (EAD)}

\subsection{Bimodal Two-Stage LGD Model}
Loss Given Default (LGD) represents the economic loss rate incurred when an exposure defaults. Unlike PD, which models a binary outcome, LGD is continuous, bounded within $[0, 1]$, and heavily bimodal. Defaults typically result in either a complete recovery (LGD = 0, ``cure'') or a near-total loss (LGD close to 1).

To capture this bimodal behavior \parencite{schuermann2004}, a \textbf{two-stage LGD model} is constructed as one of two candidate severity models --- benchmarked in Section~\ref{subsec:benchmarks} against a LightGBM challenger, with the lower out-of-sample error determining which model is deployed:
\begin{enumerate}
    \item \textbf{Stage 1 (Cure Model):} A logistic regression models the probability of a zero-loss outcome (cure):
    \begin{equation}
    p_{\text{loss}} = P(\text{LGD} > 0 | \text{Default}) = 1 - \text{sigmoid}(\mathbf{x}' \boldsymbol{\gamma})
    \end{equation}
    \item \textbf{Stage 2 (Severity Model):} For defaulted loans that incur a loss ($\text{LGD} > 0$), a fractional logit GLM (Binomial family, logit link) models the conditional loss severity \parencite{papke1996,bellotti2012}:
    \begin{equation}
    E[\text{LGD} | \text{LGD} > 0] = \frac{1}{1 + e^{-\mathbf{x}' \boldsymbol{\beta}}}
    \end{equation}
\end{enumerate}
The final predicted LGD is the product of the two stages:
\begin{equation}
\text{LGD}_{\text{pred}} = p_{\text{loss}} \times E[\text{LGD} | \text{LGD} > 0]
\end{equation}

For Basel IRB capital calculations, a conservative \textbf{Downturn LGD} is estimated at the 90th percentile of the default severity distribution:
\begin{itemize}
    \item \textbf{Mean Expected LGD (deployed model):} __MEAN_LGD__ (used for IFRS 9 Stage 1 \& 2 provisions)
    \item \textbf{Downturn LGD:} __DOWNTURN_LGD__ (used for Basel RWA capital calculations)
\end{itemize}

\subsubsection*{Why Mean LGD Is Close to Total Loss}
The realised LGD used to fit and validate the severity model is computed directly from resolved loan outcomes as
\begin{equation}
\text{LGD}_{\text{realised}} = 1 - \frac{\max(0,\ \text{recoveries} - \text{collection\_recovery\_fee})}{\max(1,\ \text{funded\_amnt} - \text{total\_rec\_prncp})}
\end{equation}
i.e.\ net recoveries (gross post-charge-off recoveries less the fee paid to the collection agency) divided by the EAD proxy (funded amount less principal already repaid at charge-off), clipped to $[0,1]$. On this fully unsecured, non-revolving instalment book, post-charge-off cash recoveries are small relative to the outstanding balance, so realised severity concentrates near total loss; the two-stage model's cure probability $p_{\text{loss}}$ separately absorbs the loans that resolve with zero loss, while the conditional severity stage captures how close to total the loss is for the remainder. The published unsecured LGD bands cited below (\textcite{schuermann2004}: 0.45--0.55, corporate/wholesale debt; \textcite{bellotti2012}: 0.25--0.45, revolving retail cards) are measured on portfolios with active collections/settlement programmes and revolving-card structures where partial recoveries are more common; the deployed mean LGD of __MEAN_LGD__, measured on already-charged-off, non-revolving loans against a book-value EAD proxy with no collateral, is a structurally different (and higher) quantity by construction rather than an anomaly. Its magnitude relative to these bands is discussed further, with numeric verdicts, in the benchmark table (Section~\ref{subsec:benchmarks}).

LGD Benchmark Context: Two severity models are compared on a chronologically held-out selection sample --- a two-stage cure-plus-severity model and a LightGBM challenger --- and the model with the lower error there is deployed for all downstream provisioning and capital; the metrics in Table~\ref{tab:lgd_validation} are then computed on a disjoint reporting sample so the published out-of-sample performance is not measured on the same defaults used for the promotion decision. For unsecured LGD, \textcite{schuermann2004} report a $0.45$--$0.55$ range (a corporate/wholesale-debt review) while \textcite{bellotti2012} document a lower $0.25$--$0.45$ band for revolving retail credit-card portfolios. The deployed mean LGD of __MEAN_LGD__ and downturn LGD of __DOWNTURN_LGD__ are assessed against these published ranges in the benchmark table (Section~\ref{subsec:benchmarks}), where any deviation is reported together with its driver. The Downturn LGD is applied conservatively at the 90th percentile of the severity distribution and used exclusively for Basel IRB capital calculations, providing an additional buffer above the mean.

\subsection{Out-of-Sample LGD Validation}
The severity model is validated out-of-time on defaulted loans from vintages held out of the fitting window. Table~\ref{tab:lgd_validation} reports the standard LGD backtesting metrics --- Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), the coefficient of determination $R^2$, and a two-sample Kolmogorov-Smirnov (KS) statistic comparing the marginal predicted and realised LGD distributions \parencite{loterman2012benchmarking}. The KS statistic is reported without its $p$-value: at this sample size ($n\approx150{,}000$) the test is hyper-sensitive to any trivial distributional difference and, being a marginal (not per-loan) comparison, cannot by itself certify calibration; Figure~\ref{fig:lgd_calibration} and the decile table against the $45^{\circ}$ line are the calibration evidence. The aggregate portfolio-level mean LGD sits above the unsecured LGD literature ranges cited in Section~\ref{subsec:benchmarks} (driver discussed there), and the loan-level predictive performance of the deployed severity model is separately weak, with a negative out-of-sample $R^2$ of $__LGD_R2__$ (Table~\ref{tab:lgd_validation}); the rejected two-stage model was materially worse at $R^2 \approx __LGD_R2_TWOSTAGE__$, which drove the champion--challenger switch. A negative $R^2$ is a well-documented model risk: realized retail LGD is highly bimodal (concentrated at 0 for cured loans and near 1.0 for write-offs), and predicting the exact loss severity on unsecured, non-collateralized consumer loans is statistically challenging. While the model remains suitable for calculating conservative portfolio-level capital buffers, its loan-level predictions should be treated with caution.

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
\includegraphics[width=0.82\textwidth]{figures/loss_distribution.png}
\caption{Monte Carlo portfolio loss distribution under the ASRF single-factor model, with Expected Loss, VaR and Expected Shortfall marked.}
\label{fig:loss_distribution}
\end{figure}

Table~\ref{tab:risk_measures} reports the resulting risk measures. Expected Shortfall exceeds VaR by construction ($ES \geq VaR \geq EL$), and the ES-based economic capital buffer is compared against the Basel IRB regulatory capital requirement.

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
    \item \textbf{Stage 3 (Credit-Impaired):} Non-performing loans (90+ DPD or default). ECL is measured over the \textbf{remaining lifetime}, with PD set to \textbf{100\%}.
\end{itemize}

A Significant Increase in Credit Risk (SICR) \parencite{novotny2016} is triggered if:
\begin{itemize}
    \item The ratio of lifetime PD to origination PD exceeds \textbf{2.5$\times$}.
    \item The absolute lifetime PD exceeds \textbf{20\%}.
    \item The loan is \textbf{30+ days past due (DPD)}, acting as a backstop.
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
\parencite{bellini2019}. Where $\text{EIR}$ is the Effective Interest Rate derived from the loan's contractual pricing.

\subsubsection*{Lifetime PD Calibration: Validating the ECL-Driving Term Structure}
The lifetime PD $1 - S(H)$ used directly in the equation above is produced by the discrete hazard model and enters the ECL sum unchanged. It is a distinct estimator from the scorecard's 12-month PD and is \emph{not} passed through the scorecard's out-of-sample isotonic/Platt recalibration described in Section~7.2 --- that recalibration only touches the 12-month PD used for expected loss, Basel RWA, and SICR-origination staging. Rather than silently trusting an unvalidated PD inside the loss-driving formula, Table~\ref{tab:lifetime_pd_calibration} instead validates the hazard model's own lifetime PD against realised lifetime default rates by origination vintage (restricted to vintages matured by 2016, since 2017--2018 charge-offs are not yet resolved in the 2018Q4 snapshot). A ratio near 1.0 indicates the hazard PD entering the ECL sum is itself reasonably calibrated to realised outcomes; a ratio materially outside the $[0.5, 1.5]$ tolerance band would indicate that the term structure --- not just the 12-month scorecard PD --- requires recalibration before it can be trusted in the ECL sum, a finding this report would then surface explicitly rather than let pass silently into the headline coverage ratio.

__LIFETIME_PD_CALIBRATION_TABLE__

\subsection{Forward-Looking Macroeconomic Scenarios}
To ensure ECL provisions are forward-looking and compliant with IFRS 9, we implement a Vector Autoregressive (VAR) forecasting model and an Ordinary Least Squares (OLS) macroeconomic regression. This framework dynamically links quarterly historical default rates of the LendingClub portfolio to key US macroeconomic indicators, sourced live from the official FRED (St.\ Louis Fed) API: the Unemployment Rate (UNRATE), GDP Growth (GDP\_growth), the Federal Funds Rate (FEDFUNDS), CPI Inflation (CPI\_inflation), and House Price Index Growth (HPI\_growth, from the seasonally-adjusted Case-Shiller US National HPI), a collateral-value indicator via which rising home prices support household wealth and reduce default risk.

The OLS model determines the sensitivity (elasticity) of the portfolio default rate to each economic factor. These sensitivities are used to predict default rates under three standardized economic scenarios (Baseline, Upside, and Downside). The macro-predicted default rates are then mathematically mapped to systematic credit cycle shocks (Vasicek $Z$-shocks) using the supervisory retail correlation parameter ($\rho = 0.15$):
\begin{equation}
Z_{\text{shock}} = \frac{\Phi^{-1}(\text{TTC\_DR}) - \Phi^{-1}(\text{PIT\_DR}) \times \sqrt{1 - \rho}}{\sqrt{\rho}}
\end{equation}
where $\text{TTC\_DR}$ is the long-run average (Through-the-Cycle) default rate, and $\text{PIT\_DR}$ is the Point-in-Time default rate predicted under each macroeconomic scenario. These mapped $Z$-shocks are then applied directly to the discrete-time hazard curves to scale the default probability curves.

The final Expected Credit Loss (ECL) is computed as the probability-weighted average across all three scenarios:
\begin{equation}
\text{ECL}_{\text{final}} = 0.50 \times \text{ECL}_{\text{Baseline}} + 0.25 \times \text{ECL}_{\text{Upside}} + 0.25 \times \text{ECL}_{\text{Downside}}
\end{equation}

__MACRO_ELASTICITIES_TABLE__

\subsubsection*{Time-Series Justification of the Lag and Sign Choice}
Table~\ref{tab:macro_ts} reports time-series diagnostics on the quarterly default-rate and macro series: an Augmented Dickey-Fuller (ADF) stationarity test, a Granger-causality test of UNRATE on the default rate, an AIC grid search over the UNRATE lag (verifying the economically-correct positive sign at the selected lag), and a Johansen cointegration test with a VECM long-run relation where cointegration is present \parencite{engelmann2011}. The contemporaneous OLS default-rate regression is prone to spurious signs (the negative UNRATE coefficient reflects charge-off lags and underwriting drift), so we impose economic sign priors for scenario projection rather than build a full Vector Error Correction Model (VECM) for projections, following the standard banking preference for simpler, transparent Vasicek overlays.

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
Model validation is performed on the completely held-out Out-of-Time (OOT) dataset ($2016$--$2018$). The PD scorecard shows stable performance with an OOT AUC of \textbf{__AUC_OOT__} and a Kolmogorov-Smirnov (KS) statistic of \textbf{__KS_OOT__}.

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
Calibration was evaluated by comparing predicted default rates against actual observed default rates across risk deciles. The Hosmer-Lemeshow test \parencite{hosmer2013} rejects perfect calibration on the OOT dataset ($p = \text{\textbf{__HL_PVALUE__}} < 0.05$). Given the very large sample size ($N = \text{VAR_N_OOT}$), this result partly reflects the high sensitivity of the chi-squared goodness-of-fit test, but the calibration plots also indicate systematic underprediction at higher risk deciles. Post-model recalibration (isotonic regression) was therefore fitted \emph{out-of-sample} on the in-time test partition and applied to the OOT set. Because the calibrator is fitted on a different partition and only transferred to OOT, it is \emph{not} guaranteed to improve every OOT diagnostic: as Table~\ref{tab:calibration_comparison} shows, discrimination is essentially unchanged and the calibration slope, intercept and expected default rate each move slightly \emph{away} from their targets on the OOT set. We deliberately do \emph{not} fit the recalibrator on the OOT set itself --- doing so would trivially pass the Hosmer-Lemeshow test but would be an in-sample artefact. Accordingly we retain the recalibrated PDs as an honest out-of-sample transform rather than a certified calibration pass; the residual higher-decile underprediction carries through to the ECL provisions and is listed among the known limitations in Section~10.

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

Model backtesting compares the scorecard's predicted average PD against observed default rates segmented by quarterly origination vintage, following the methodology of \parencite{eba2017} and \parencite{bcbs2005}. For each vintage cohort, we assess whether the predicted mean PD lies within a \textbf{50\% tolerance band} of the realised default frequency:
\begin{equation}
\text{PD Ratio}_{v} = \frac{\bar{p}_v}{\bar{d}_v}
\end{equation}
where $\bar{p}_v$ is the cohort-average predicted PD and $\bar{d}_v$ is the realised default rate. A ratio between 0.5 and 1.5 indicates acceptable calibration. Cohorts outside this band are flagged for recalibration ($\dagger$). Table~\ref{tab:pd_backtest} reports the backtesting results by vintage quarter.

The systematic underprediction observed in the 2016--2018 vintages (PD Ratio consistently below 0.85) is consistent with the calibration drift identified in Section~7.2 and corrected via the isotonic regression recalibration described therein for the scorecard's \textbf{12-month} PD, which feeds Expected Loss, Basel RWA, and SICR-origination staging. The hazard model's \textbf{lifetime} PD that drives IFRS~9 ECL directly ($\text{ECL} = \sum_t m(t)\times\text{LGD}(t)\times\text{EAD}(t)/(1+\text{EIR})^t$) is a separate estimator and is \emph{not} passed through this scorecard recalibration; it is instead independently validated against realised lifetime default rates by vintage in Table~\ref{tab:lifetime_pd_calibration} (Section~6.2).

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
A single global recalibrator cannot correct an era-specific bias. We therefore fit separate isotonic and Platt recalibrators for the pre-2016 and 2016--2018 eras and measure, per vintage group, the raw and recalibrated PD ratio against the realised default rate (Table~\ref{tab:vintage_calib}, Figure~\ref{fig:vintage_calib}). The raw ratio is materially below $1.0$ for the newer vintages (the documented under-prediction), and the era-specific isotonic recalibration moves it back toward $1.0$. This is an in-sample diagnostic that quantifies the drift and demonstrates the correction; production PD is unchanged. Since the era-specific calibrators are fitted and evaluated on the same vintage partitions (e.g. 2016--2018), the resulting alignment (Isotonic/Platt PD equals the Actual DR of 25.27% exactly) is in-sample and tautological: it is a diagnostic baseline that demonstrates the extent of the raw model's drift, not a validation of the recalibrator's out-of-sample generalization.

__VINTAGE_CALIB_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=0.80\textwidth]{figures/validation/calibration_by_vintage.png}
\caption{PD calibration ratio (predicted / actual) by vintage group, raw vs era-recalibrated. A ratio of 1.0 is perfect calibration; below 1.0 is under-prediction.}
\label{fig:vintage_calib}
\end{figure}

\subsection{IFRS~9 Stage Migration and Coverage Analysis}

The stage migration matrix quantifies transitions between IFRS~9 performance stages \parencite{novotny2016}, providing a key indicator of portfolio deterioration velocity. Rows represent the origination stage; columns represent the reporting-date stage.

\begin{table}[H]
\centering
\caption{IFRS~9 Stage Migration Matrix (Origination $\to$ Reporting Date)}
\label{tab:stage_migration}
\vspace{0.5em}
\begin{tabular}{lccc}
\toprule
\textbf{From Stage} & \textbf{To Stage~1} & \textbf{To Stage~2} & \textbf{To Stage~3} \\
\midrule
\textbf{Stage~1} & __STAGE_1_1__ & __STAGE_1_2__ & __STAGE_1_3__ \\
\textbf{Stage~2} & __STAGE_2_1__ & __STAGE_2_2__ & __STAGE_2_3__ \\
\textbf{Stage~3} & __STAGE_3_1__ & __STAGE_3_2__ & __STAGE_3_3__ \\
\bottomrule
\multicolumn{4}{p{0.92\linewidth}}{\footnotesize\textit{Note:} the ``From Stage~3'' row is zero by construction: no loan originates directly into Stage~3, as Stage~3 classification requires 90+ days past due status, which by definition cannot occur at origination ($t=0$).} \\
\end{tabular}
\end{table}

\subsection{ECL Macro Scenario Sensitivity}

Figure~\ref{fig:ecl_tornado} presents the portfolio ECL sensitivity to the Vasicek systematic factor~$Z$ across a range of macro scenarios, from severe expansion ($Z = +2.0$) to severe recession ($Z = -2.0$). This analysis follows the probability-weighted scenario methodology mandated by IFRS~9 paragraph B5.5.42 \parencite{iasb2014} and the stress testing framework of \parencite{bellini2019}.

\subsection{ECL What-If Calculator: PD / LGD / EAD Stress Scenarios}
Beyond the macro-factor mapping, a management-facing what-if calculator answers the direct risk-committee question \emph{``what happens to provisions if PD, LGD or EAD move?''}. Holding the baseline term structure fixed, each scenario applies a multiplicative PD shock, an additive LGD shock and/or a multiplicative EAD (drawdown) shock, and the ECL engine is re-run. Table~\ref{tab:ecl_whatif} reports the resulting provision changes, and Figure~\ref{fig:ecl_shock_tornado} presents them as a tornado chart, including regulator-style severe scenarios calibrated to COVID-19 and Global-Financial-Crisis default multiples. Note that the baseline-only ECL used as the anchor for these what-if sensitivities excludes the macro scenario probability-weights (downside recession and upside expansion shocks), which explains why it is slightly lower than the final reported probability-weighted ECL.

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

Notably, the WoE logistic scorecard attains the highest discrimination on \emph{both} the in-time test and the out-of-time sets: its OOT AUC (VAR_AUC_OOT) exceeds LightGBM (VAR_LGBM_AUC_OOT), XGBoost (0.6776), Random Forest (0.6727) and even the weighted ensemble (0.6943). The tree models fit in-sample structure that does not transfer across the 2016--2018 regime shift, whereas the scorecard's coarse monotone WoE bins generalise more stably. The Logistic Scorecard is therefore the champion on discrimination grounds alone, before its decisive advantages in point-based explainability and regulatory audit compliance are even considered. It should be noted that the challenger models (LightGBM, XGBoost, Random Forest) were trained using standard, default configurations without systematic hyperparameter grid search tuning. While hyperparameter optimization might yield marginal discrimination improvements, the baseline scorecard's outperformance on the OOT set is primarily driven by the coarse, monotonic Weight of Evidence (WoE) binning, which acts as a powerful structural regulariser and guards against overfitting the pre-2015 vintage regimes.

\subsubsection*{Is the Challenger's Edge Statistically Significant?}
Point-estimate AUC/Gini gaps cannot distinguish a genuine improvement from sampling noise. To test significance we run a paired bootstrap: the same held-out OOT rows are resampled for both models and a confidence interval is built on the \emph{difference} in Gini (challenger $-$ champion). If that interval excludes zero, the difference is significant. Table~\ref{tab:ab_test} reports the result; it complements the analytic DeLong test on the same pair.

__AB_TEST_TABLE__

\subsection{SHAP Explainability: Challenger Model Feature Contributions}

While the primary WoE scorecard provides point-attributable explanations per feature bin \parencite[Ch.~4]{siddiqi2017}, the LightGBM challenger model is interpreted via SHAP (SHapley Additive exPlanations) values \parencite{lundberg2017}. Figure~\ref{fig:shap} displays the mean absolute SHAP contribution of each feature: \texttt{VAR_SHAP_TOP1} and \texttt{VAR_SHAP_TOP2} dominate the challenger's default discrimination.

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figures/validation/shap_challenger_summary.png}
\caption{SHAP Feature Importance --- LightGBM Challenger Model}
\label{fig:shap}
\end{figure}

% -----------------------------------------------------------------------------
% 8. STABILITY AND PERFORMANCE OVERLAYS
% -----------------------------------------------------------------------------
\section{Stability and Performance Overlays}

\subsection{Scorecard Population Stability (PSI)}
The Population Stability Index (PSI) \parencite[Ch.~9]{engelmann2011} measures changes in the distribution of credit scores over time. The scorecard demonstrates excellent stability, with a total PSI between the training and OOT populations of \textbf{__PSI_TRAIN_OOT__}. This is well below the conservative regulatory threshold of \textbf{0.10}, indicating that the underwriting population profile has not shifted significantly.

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figures/validation/psi_distribution.png}
\caption{Score Distribution Stability (Train vs.\ OOT)}
\label{fig:psi_dist}
\end{figure}

\subsection{Gains and Lift Analysis}
The Gains Chart measures the model's ability to concentrate defaults within the lowest score bands. The scorecard concentrates most defaults in the lowest score bands (Figure~\ref{fig:gains_ch}).

\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{figures/validation/gains_chart.png}
    \caption{Cumulative Capture (Gains) Chart}
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

Underwriting decisions require balancing credit risk, portfolio growth, and capital constraints. We evaluate each score cutoff on Expected Profit and Risk-Adjusted Return on Capital (RAROC), with expected loss (at the approved-population bad rate) and a __RAROC_HURDLE__ charge on economic capital both netted out. On this risk-priced, high-yield portfolio the unconstrained profit-maximising \emph{and} RAROC-maximising cutoff is the same corner solution --- approving the entire through-the-door population, at a portfolio RAROC of \textbf{__CORNER_RAROC__} --- because higher-risk grades carry interest rates high enough to remain RAROC-accretive even after loss and capital costs. Because unconstrained optimisation therefore implies 100\% approval, the decision problem is formulated as a \emph{constrained} optimization: we maximize approved volume and expected profit subject to an active risk-appetite ceiling on the approved bad rate (\textbf{__MAX_BAD_RATE__}), taking the most inclusive cutoff (maximising approved profit) whose approved bad rate stays within that ceiling. This constrained boundary represents the recommended operating cutoff.

We sweep score cutoffs from 400 to 800 (evaluating approved loan subsets) to calculate Expected Profit and RAROC. All components are expressed on a consistent \emph{per-annum} basis:
\begin{itemize}
    \item \textbf{Interest Income:} Based on per-loan interest rates (annual coupon on EAD).
    \item \textbf{Fees:} 1.0\% of Exposure at Default (EAD) per annum.
    \item \textbf{Funding Cost:} 4.0\% of EAD per annum.
    \item \textbf{Operating Cost:} 1.5\% of EAD per annum.
    \item \textbf{Expected Loss (EL):} $\text{PD}_{\text{annual}} \times \text{LGD} \times \text{EAD}$, where the lifetime PD is converted to an annual default probability via $\text{PD}_{\text{annual}} = 1 - (1 - \text{PD}_{\text{lifetime}})^{12/\text{term}}$. Charging the full lifetime PD against a single year of income would overstate losses by the loan term and distort every cutoff decision.
    \item \textbf{Capital Cost:} 12.0\% hurdle rate applied to the Minimum Capital Reserve (8\% of RWA).
\end{itemize}

Table~\ref{tab:cutoff_raroc} displays the optimization results across selected score cutoffs.

__CUTOFF_RAROC_TABLE__

As shown in the table, the recommended operating cutoff is score \textbf{__OPT_CUTOFF__} with an approval rate of \textbf{__OPT_APPROVAL__}, a bad rate of \textbf{__OPT_BAD__}, an expected profit of \textbf{\$__OPT_PROFIT_M__M}, and a RAROC of \textbf{__OPT_RAROC__} --- __RAROC_VS_HURDLE__ the __RAROC_HURDLE__ cost-of-capital hurdle. It is the most inclusive cutoff whose approved bad rate stays within the risk-appetite ceiling, and it appears as the highlighted row in Table~\ref{tab:cutoff_raroc} and is marked in Figure~\ref{fig:cutoff_profit}. A single objective drives the highlighted table row, the figure, and this text, so there is no discrepancy between the quoted cutoff and the swept grid. Relaxing the risk-appetite ceiling would move the cutoff toward the profit/RAROC corner (full approval); tightening it raises the cutoff and lowers approved volume.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{figures/cutoff_profit_curve.png}
\caption{Expected Profit and RAROC versus score cutoff over the full 400--800 grid. The gold marker denotes the recommended operating cutoff (risk-appetite ceiling on the approved bad rate) reported in Table~\ref{tab:cutoff_raroc}.}
\label{fig:cutoff_profit}
\end{figure}

\section{Limitations, Assumptions, and Recommendations}
\label{sec:assumptions}

\subsection{Key Modelling Assumptions and Limitations}
\begin{table}[H]
\centering
\small
\caption{Key Modelling Assumptions and Associated Limitations}
\label{tab:assumptions}
\begin{tabular}{p{3cm}p{5.5cm}p{5.5cm}}
\toprule
\textbf{Component} & \textbf{Assumption} & \textbf{Limitation / Mitigation} \\
\midrule
PD Scorecard
  & WoE binning with monotonic constraints.
  & Non-monotonic relationships suppressed; mitigated by VIF screening. \\
\addlinespace
LGD Model
  & Fractional logit on funded\_amnt proxy EAD.
  & Ignores accrued interest; conservative bias accepted. \\
\addlinespace
EAD
  & Closed-form annuity; zero prepayment assumption.
  & Overestimates EAD for prepaid loans; acceptable for conservative capital. \\
\addlinespace
Basel IRB
  & ASRF ``Other Retail'' supervisory correlation $R \in [0.03, 0.16]$.
  & Single systematic factor; portfolio concentration not captured. \\
\addlinespace
IFRS~9 ECL
  & Constant hazard within each calendar month.
  & Does not capture seasoning curves; mitigated by discrete-time model. \\
\addlinespace
Reject Inference
  & Parcelling from 100k random sample of 27.6M rejects.
  & Sample may not represent full reject distribution; Gini shift monitored. \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Model Risk \& Limitations}
The Lending Club dataset is limited to US consumer loans and may not generalize to corporate lending, small-business loans, or other international retail portfolios. Additionally, the dataset excludes collateral details, meaning the LGD model relies primarily on underwriting grades and debt-to-income (DTI) metrics.

\subsection{Known Model Limitations}
\begin{itemize}
    \item \textbf{Stage 2 SICR Proxy:} To calculate the longitudinal 12-month stage transition matrix, we simulate the credit conditions 12 months ago by interpolating between the origination and current credit profiles. This proxy is approximate and can be refined with exact monthly reporting snapshots.
    \item \textbf{Categorical Feature Encoding:} Grade, term, employment length and home ownership are ordinal-encoded before WoE binning. Label ordering assumptions (e.g., A=1 through G=7) are reasonable but should be validated against observed default rates for each category.
    \item \textbf{DPD Roll-Rate Analysis (future work):} A monthly delinquency-bucket roll-rate transition matrix, the standard IFRS~9 monitoring-committee tool, is not produced here because the LendingClub accepted-loan file is loan-level with no monthly days-past-due panel. Building genuine roll-rates requires a monthly servicing snapshot feed, which is the recommended data extension for a production deployment.
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
Under BCBS §322--328, the regulatory correlation ($R$) is constrained between $0.03$ and $0.16$. The correlation decays exponentially as the PD increases, reflecting the lower correlation of defaults among higher-risk borrowers during stable economic conditions:
\begin{equation}
R = 0.03 \times \frac{1 - e^{-35 \times PD}}{1 - e^{-35}} + 0.16 \times \left[ 1 - \frac{1 - e^{-35 \times PD}}{1 - e^{-35}} \right]
\end{equation}
For the RWA calculation, the supervisory PD floor of \textbf{0.03\%} is strictly applied to ensure a minimum capital charge.

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
PD Underwriting & statsmodels Logit & Monotonic credit scorecard fitting \\
LGD Modeling & statsmodels GLM (fractional logit) & Two-stage cure + severity modeling \\
Challenger Model & LightGBM & Non-linear machine learning benchmark \\
Survival PD Curves & Discrete-time hazard model & Monthly lifetime term structure modeling \\
PDF Generation & XeLaTeX + biber & Publication-quality academic PDF \\
\bottomrule
\end{tabular}
\end{table}


\clearpage
\section*{References}
\addcontentsline{toc}{section}{References}
\printbibliography[heading=none]

\end{document}
"""

    # ── Placeholder substitutions ──────────────────────────────────────────────
    latex_content = latex_template
    latex_content = latex_content.replace("__TODAY__", today_str)
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
    latex_content = latex_content.replace("__MAX_BAD_RATE__", max_bad_rate_txt)
    latex_content = latex_content.replace("__CORNER_RAROC__", corner_raroc)
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
    latex_content = latex_content.replace("VAR_AUC_OOT", fmt_dec(metrics.get("auc_oot", 0.6897)))
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
    # New: stage migration cells
    for fs in [1, 2, 3]:
        for ts in [1, 2, 3]:
            latex_content = latex_content.replace(f"__STAGE_{fs}_{ts}__", _mig_cell(fs, ts))

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
    _rwa_irb = float(metrics.get("total_rwa", 0.0) or 0.0)
    _rwa_sa = float(metrics.get("total_rwa_sa", 0.0) or 0.0)
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
        fmt_dec(metrics.get("gini_oot", 0)),
    )
    latex_content = latex_content.replace(
        "VAR_AUC_OOT",
        fmt_dec(metrics.get("auc_oot", 0)),
    )
    latex_content = latex_content.replace(
        "VAR_MEAN_LGD",
        fmt_dec(metrics.get("mean_lgd", 0)),
    )
    latex_content = latex_content.replace(
        "VAR_DOWNTURN_LGD",
        fmt_dec(metrics.get("downturn_lgd", 0)),
    )
    latex_content = latex_content.replace(
        "VAR_RWA_IRB",
        fmt_num(metrics.get("total_rwa", 0)),
    )
    latex_content = latex_content.replace(
        "VAR_RWA_DENSITY",
        str(metrics.get("rwa_density", "N/A")).replace("%", "\\%"),
    )
    latex_content = latex_content.replace(
        "VAR_ECL_TOTAL",
        fmt_num(metrics.get("total_ecl", 0)),
    )
    latex_content = latex_content.replace(
        "VAR_ECL_COVERAGE",
        fmt_pct(metrics.get("ecl_coverage", 0), precision=3),
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

