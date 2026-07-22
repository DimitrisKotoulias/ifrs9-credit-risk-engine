# Credit Risk & IFRS 9 ECL Pipeline and Report Enhancements Design Spec

This document specifies the technical design and architectural enhancements for the Credit Risk & IFRS 9 ECL engine. The modifications implement a selection bias correction using Reject Inference (Parcelling) and a mathematically rigorous Point-in-Time credit cycle hazard adjustment using the Vasicek Single Factor Model, alongside Basel III Economic Capital stress-testing. Finally, the Model Risk Management (MRM) report is personalized and upgraded to a premium investment-banking deliverable.

---

## 1. User Customization & Aesthetics
To personalize the deliverable for credit risk validation reviewers:
- **Author Attribution:** The report cover page will prominently list **Dimitrios Kotoulias** as the lead Quantitative Validation Analyst.
- **Title Grid Alignment:** The cover page metadata block will be transformed into an elegant, 4-column balanced grid to present "Prepared By", "Data Source", "Report Date", and "Model Version" side-by-side.
- **Premium Styling rules:** Sleek borders, high-contrast typography using Inter and Cinzel fonts, custom CSS page headers/footers, and clean page-break alignments for nested tables and charts.

---

## 2. Option A: Reject Inference Integration (Parcelling)

### 2.1 Context and Selection Bias
A fundamental challenge in retail credit scorecard development is **selection bias**. Scorecard models are traditionally fitted only on approved applications (where credit default outcomes $Y \in \{0,1\}$ are observed). Applicants who were rejected have unobserved outcomes. A scorecard fitted solely on accepted loans underestimates risk for low-scoring applicants and degrades model performance when applied to the "through-the-door" (all applicants) population.

### 2.2 Parcelling Mathematical Formulation
To resolve this, we implement the **Parcelling** reject inference technique:
1. Train an initial "Accepts-Only" scorecard model on the accepted population.
2. Align the features of the rejected population (imputing missing post-origination fields and mapping variable names) and generate predicted default probabilities $P(\text{Bad} | \mathbf{x}_{\text{rej}})$ from the accepts scorecard.
3. For each rejected loan $i$, create two fractional pseudo-records:
   - **Good Pseudo-Record:** $Y_i = 0$, with weight $w_{i,G} = 1 - P(\text{Bad} | \mathbf{x}_{\text{rej}})$
   - **Bad Pseudo-Record:** $Y_i = 1$, with weight $w_{i,B} = P(\text{Bad} | \mathbf{x}_{\text{rej}})$
4. Pool the original accepted records (each assigned weight $w = 1.0$) with both sets of rejected pseudo-records.
5. Refit the logistic scorecard on the pooled dataset using sample weights.
6. Calculate the **Gini Coefficient Shift** to measure selection bias correction:
   $$\text{Gini Shift} = \text{Gini}_{\text{through-door}} - \text{Gini}_{\text{accepts}}$$

### 2.3 Feature Mapping & Imputation
Since the rejected Lending Club dataset has a restricted schema compared to the accepted one, we perform the following mapping:
- `risk_score` $\rightarrow$ `fico_range_low`, `fico_range_high` (`fico_high = risk_score + 4`)
- `debt_to_income_ratio` $\rightarrow$ `dti`
- `employment_length` $\rightarrow$ `emp_length`
- `loan_amnt` $\rightarrow$ `loan_amnt`
- `annual_inc` $\rightarrow$ `annual_inc`
- Non-overlapping scorecard features (e.g., `int_rate`, `delinq_2yrs`, `open_acc`, `pub_rec`) will be imputed using conservative segment-level modes or overall population means from training.

---

## 3. Option B: Vasicek Credit Cycle Model & Capital Stress Testing

### 3.1 Vasicek Single Factor Model
Under the Basel Asymptotic Single Risk Factor (ASRF) framework, the systematic macroeconomic credit cycle state is modeled using a standard normal systematic risk factor $Z \sim \mathcal{N}(0,1)$, representing the health of the economy (where a higher $Z$ indicates a strong economy and a lower $Z$ represents recession).

The Point-in-Time (PiT) Probability of Default $PD_{PiT}(Z)$ is linked to the Through-the-Cycle (TTC) base probability $PD_{TTC}$ via the **Vasicek Model**:
$$PD_{PiT}(Z) = \Phi\left( \frac{\Phi^{-1}(PD_{TTC}) - \sqrt{\rho} Z}{\sqrt{1 - \rho}} \right)$$
where:
- $\Phi$ is the standard normal cumulative distribution function.
- $\Phi^{-1}$ is the inverse standard normal cumulative distribution function.
- $\rho$ is the systematic asset correlation (calibrated from regulatory retail formulas or configured).
- $Z$ is the macroeconomic systematic shock factor.

### 3.2 Term Structure Dynamic Scaling
We will replace the arbitrary log-odds shift in `DiscreteHazardModel` with the Vasicek formula:
1. For a systematic macro factor shock $Z$, the baseline conditional default hazards $h(t)$ are transformed:
   $$h_{PiT}(t | Z) = \Phi\left( \frac{\Phi^{-1}(h(t)) - \sqrt{\rho} Z}{\sqrt{1 - \rho}} \right)$$
2. Project probability-weighted survival curves $S(t | Z)$ and marginal default rates across:
   - **Baseline Scenario:** $Z = 0.0$
   - **Upside Scenario:** $Z = 1.0$ (Expansion)
   - **Downside Scenario:** $Z = -1.5$ (Contraction)
3. The dynamic staged ECL is then computed using these probability-weighted marginal default rates.

### 3.3 Basel III Unexpected Loss Capital Stress Test
Under a severe economic recession scenario ($Z = -2.0$), we will compute stressed unexpected losses (UL):
- **Stressed PD:** Compute $PD_{\text{stress}}$ using the Vasicek formula at $Z = -2.0$.
- **Stressed Capital Requirement ($K_{\text{stress}}$):** Plug $PD_{\text{stress}}$ and Downturn LGD into the Basel III retail capital formula.
- **Stressed RWA:** $RWA_{\text{stress}} = K_{\text{stress}} \times 12.5 \times EAD$.
- Compare the regulatory capital reserves required under standard vs. stressed conditions to establish capital adequacy buffers.

---

## 4. Proposed Changes by File

### 4.1 `pipeline.py`
- Modify the return capture of `load_and_prepare` to store `df_rejected`.
- Add **Phase 9b: Reject Inference & Refitting**. Align, impute, and score `df_rejected`. Run `refit_with_parcelling` from `reject_inference.py`. Write `gini_shift` and new through-the-door metrics to `metrics.json`.
- Add **Phase 9c: Basel Economic Capital Stress Test**. Calculate stressed PDs, stressed Basel K, RWA, and capital requirements at $Z = -2.0$. Save stressed metrics to `metrics.json`.

### 4.2 `pd_term_structure.py`
- Implement the Vasicek transformation in `_hazard_at_t` and `predict_term_structure`. Add asset correlation $\rho$ parameter to `__init__` (default $0.15$).

### 4.3 `ifrs9_ecl.py`
- Update macro shock calculations to align with systematic factor $Z$ draws.
- Propagate Vasicek weights to Baseline, Upside, and Downside.

### 4.4 `basel_irb.py`
- Expose a helper function to calculate the capital requirement $K$ and $RWA$ for a given $PD$ array to enable rapid stress testing.

### 4.5 `report.html.j2`
- Add **Dimitrios Kotoulias** to the title page metadata.
- Upgrade `.title-meta` layout CSS.
- Add Section 6b: **"Basel III Economic Capital & Macro Stress Testing"** with the Vasicek model formulation, LaTeX equations, and comparison tables.
- Add Section 9b: **"Selection Bias & Reject Inference (Parcelling)"** with Gini comparison tables and methodology.

---

## 5. Verification Plan
1. Run end-to-end execution of `python -m credit_risk.pipeline` to generate `outputs/metrics.json` containing reject inference and stress testing results.
2. Verify that `pytest` passes all validation, data, and risk mathematical tests.
3. Render the HTML/PDF report via `python -m credit_risk.reporting.render` and visually inspect that the styling, covers, and tables are formatted perfectly.
