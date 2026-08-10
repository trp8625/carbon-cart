# Carbon Footprint Nudge: An A/B Testing Case Study

*A simulated e-commerce checkout experiment combining real-world carbon emission data (Climatiq API) with a full randomized experimentation pipeline: power analysis, CUPED variance reduction, sequential testing correction, multi-variant FDR correction, and a causal-inference comparison against observational data.*

All numbers below come from actual runs of the project's scripts, not hand-picked examples. Effect sizes, baseline rates, and other simulation parameters were assumptions set by the author, clearly labeled as such in the code -- the point of the project is the experimentation methodology, not a claim about real consumer behavior.

See the [README](../README.md) for a quick-start overview; this document is the full technical write-up.

---

## Executive Summary

A simulated checkout experiment tested whether showing shoppers a real-time carbon footprint estimate for their cart shifts their behavior. Across 16,800 simulated users, the nudge produced a statistically significant 2.45-percentage-point drop in conversion (p = 0.0006) and a statistically significant 8.82-percentage-point increase in selecting a lower-carbon alternative among people who did convert (p < 0.0001). However, the guardrail metric -- total carbon footprint per user shown checkout, accounting for the conversion loss -- did not move significantly (p = 0.39), meaning the conversion cost and the greener-choice benefit roughly offset each other in aggregate. A parallel causal-inference exercise showed that if this same effect were estimated from biased observational data instead of a proper randomized experiment, naive analysis would have concluded the *opposite* sign of effect entirely -- underscoring why the randomized design matters.

---

## 1. Experiment Design

**Setup.** Simulated users arrive at checkout with a cart (1 of 5 product categories, lognormal cart value centered around $60). Users are randomly assigned 50/50 to control (no carbon information shown) or treatment (a carbon footprint estimate is shown, with the option to switch to a lower-carbon alternative). Two outcomes are tracked: whether the user completes the purchase (`converted`), and, among those who convert, whether they chose the lower-carbon alternative (`chose_green_alt`).

**Assumed effects** (author-set, not measured): the nudge introduces checkout friction (-2 percentage points on conversion) but increases green-alternative selection among converters (+10 percentage points). A secondary, smaller effect ties historical spending to conversion probability, used later as the covariate for CUPED.

**Sample size.** A power analysis (Section 2) targeting an ability to detect a 2-percentage-point conversion change at 80% power and alpha = 0.05 indicated ~8,392 users per group (16,785 total). The simulation used 16,800 users, matching that target.

---

## 2. Data Sources & Power Analysis

### 2.1 Real Emission Factors (Climatiq API)

Category-level carbon intensity (kg CO2e per $1 of spend) was pulled from Climatiq's free-tier API rather than invented, so that the "sustainability" side of the guardrail metric reflects real published emission data:

| Category | Factor source | Region | kg CO2e per $1 |
|---|---|---|---|
| Apparel | Apparel manufacturing | TN | 1.0139 |
| Home goods | Institutional furniture manufacturing | DK | 1.4087 |
| Food & beverage | Food and beverage stores | US | 0.2760 |
| Books/media | Periodical publishers | US | 0.0590 |
| Electronics | *No money-based match found* | -- | 0.45 (placeholder fallback) |

Electronics is a known gap: the automated search query didn't return a spend-based emission factor for that category, so the simulation falls back to a placeholder value for it specifically. This is flagged explicitly in the script output and is the one part of the pipeline not backed by a real, sourced emission factor -- worth revisiting (e.g. a more targeted search query, or manually selecting an activity ID via Climatiq's Data Explorer) before treating category-level footprint comparisons as fully defensible.

### 2.2 Power Analysis

Using a two-proportion z-test framework (`statsmodels.stats.power.NormalIndPower`), required sample size per group at alpha = 0.05, power = 0.80, baseline conversion = 70%:

| Minimum detectable effect | n per group | Total n |
|---|---|---|
| 1.0 pp | 33,273 | 66,546 |
| 2.0 pp | 8,392 | 16,785 |
| 3.0 pp | 3,762 | 7,523 |
| 5.0 pp | 1,376 | 2,751 |

The 2pp target was chosen as the design threshold, and the actual observed effect (2.45pp) was detectable at that sample size, as confirmed in Section 3.

---

## 3. Primary A/B Test Results

| Metric | Control | Treatment | Difference | 95% CI | p-value | Result |
|---|---|---|---|---|---|---|
| Conversion rate | 0.7026 (n=8,446) | 0.6781 (n=8,354) | -0.0245 | [-0.0384, -0.0105] | 0.0006 | **Significant** |
| Green-alt selection (among converters) | 0.1582 (n=5,934) | 0.2464 (n=5,665) | +0.0882 | [+0.0736, +0.1028] | <0.0001 | **Significant** |
| Effective CO2e/user shown checkout (kg) | 38.845 | 38.135 | -0.710 | [-2.331, +0.911] | 0.3907 | Not significant |

The "effective CO2e" metric counts a cart's footprint only if the purchase was completed (an abandoned cart has no real-world emissions), so it reflects the combined effect of the conversion drop and the greener choices among those who did convert.

**Reading this together:** the nudge reliably works as intended -- it makes people who buy choose lower-carbon options far more often -- but it also reliably costs some conversions, and the net effect on total footprint is statistically indistinguishable from zero. The two forces are pulling in the same direction on the guardrail metric (fewer purchases *and* greener purchases both reduce footprint) but neither is large enough, nor is the sample large enough, to detect a combined effect with confidence at this sample size.

---

## 4. CUPED Variance Reduction

CUPED (Deng, Xu, Kohavi & Walker, 2013 -- verify this citation independently if used elsewhere) reduces variance in an experiment's outcome metric using a pre-experiment covariate. Here, `historical_avg_spend` was used as the covariate for the conversion metric.

- Correlation between conversion and historical spend: 0.0821 (weak)
- Variance reduction achieved: 0.67%
- Estimated effect without CUPED: -0.02446 (p = 0.0006)
- Estimated effect with CUPED: -0.02462 (p = 0.0005)

The variance reduction was minimal, which is the expected and correct outcome given the weak covariate correlation -- CUPED's benefit is bounded by how strongly the chosen covariate actually predicts the outcome, and historical spend just isn't a strong predictor of conversion in this simulation. The two effect estimates are nearly identical, as they should be (CUPED doesn't change what's being estimated, only how precisely). This is a useful negative result for the write-up: it demonstrates the mechanism correctly and illustrates that covariate selection matters -- a stronger covariate (e.g., a user's known purchase intent score, if available) would likely have produced a larger reduction.

---

## 5. Sequential Testing / Peeking Correction

This section simulates what happens if results are checked repeatedly as data accumulates (e.g., daily), rather than only once at a fixed sample size.

**The problem, quantified:** simulating 3,000 null experiments (no true treatment effect) with 10 interim looks each, a naive rule of "stop as soon as any interim p < 0.05" produced a false-positive rate of **20.3%** -- more than four times the nominal 5% alpha.

**The correction:** rather than relying on a textbook Pocock/O'Brien-Fleming formula, a corrected critical value was calibrated empirically via the same Monte Carlo simulation, finding the |z| threshold that keeps the any-look false-positive rate at 5%: **|z| > 2.582**, versus the uncorrected 1.960.

**Applied to the real experiment:** on the actual data, a naive rule would have stopped and declared significance at look 2 (z = -2.668). Reassuringly, that same look also clears the corrected threshold (2.668 > 2.582) -- so in this case, the conclusion holds up under valid sequential testing. This is worth stating plainly: the *existence* of a peeking problem doesn't mean every naive-looking result is wrong, only that the naive false-positive rate is inflated in general; each specific result still needs to be checked against the corrected bar, which this one passes.

---

## 6. Multi-Variant Testing & FDR Correction

Three nudge variants were tested simultaneously against control: a plain carbon estimate, the estimate plus a one-click greener alternative, and the estimate plus social-proof framing.

| Variant | Conversion rate | Green-alt rate (among converters) |
|---|---|---|
| Control | 0.6952 | 0.1493 |
| Plain estimate | 0.6958 | 0.2083 |
| Green alternative | 0.6895 | 0.2662 |
| Social proof | 0.6974 | 0.2278 |

None of the three variants showed a statistically significant conversion difference from control, naively or after Benjamini-Hochberg FDR correction (all p-values between 0.57 and 0.96).

**Why:** splitting the same 16,800-user pool across 4 arms instead of 2 leaves roughly 4,200 users per arm -- well under the ~8,400 per group the Section 2 power analysis said was needed to detect a 2pp effect. This connects directly back to the power analysis: testing more variants simultaneously isn't free -- it costs statistical power unless total traffic is scaled up to compensate. No FDR-driven "flip" from naive-significant to corrected-not-significant occurred in this run, simply because nothing reached naive significance to begin with; the underpowering, not the correction itself, is the binding constraint here.

---

## 7. Causal Inference Comparison

**Motivation.** The core experiment above uses proper randomization, the gold standard for causal effect estimation. Most real-world product data isn't collected this way -- features often roll out to self-selected or targeted groups. This section asks: if this same effect had to be estimated from observational (non-randomized) data instead, how far off would the naive conclusion be, and can standard adjustment techniques close the gap?

**Setup.** A second, independent dataset was simulated where seeing the carbon nudge is *not* randomly assigned. Instead, a latent "eco-consciousness" trait drives both who saw the nudge (more eco-conscious users self-selected in) and conversion behavior directly -- this is a classic confounder. The true causal effect was fixed at -0.02 (matching the randomized experiment's assumed effect), a figure only knowable because the data is simulated.

Three estimates of the same effect were computed: a naive difference in means, propensity-score matching (logistic regression for propensity scores + nearest-neighbor matching, implemented directly with scikit-learn), and the actual RCT result as ground truth.

| Method | Estimated effect | Bias vs. true effect (-0.0200) |
|---|---|---|
| True effect (known, by simulation design) | -0.0200 | -- |
| Naive observational | +0.0161 | +0.0361 |
| Propensity-score matched | -0.0293 | -0.0093 |
| Randomized (RCT) | -0.0245 | -0.0045 |

PSM 95% CI on the matched effect: [-0.0429, -0.0157] (contains the true effect). Mean propensity-score distance between matched pairs: 0.0002 (close matches).

**Interpretation.** The naive comparison doesn't just add noise -- it reverses the sign of the effect. Because eco-conscious users convert at a higher baseline rate *and* were far more likely to have seen the nudge, the confounding overwhelms and flips the true (negative) effect into an apparently positive one. Propensity-score matching, which compares users with similar odds of having seen the nudge rather than the full unmatched population, recovers an estimate close to the true effect and correctly signed -- though not identical to the RCT result, since matching can only adjust for confounders that were actually measured. The RCT remains the most accurate of the three: randomization removes confounding by design, without requiring anyone to correctly guess which variables need adjusting for after the fact.

**Limitation.** This comparison uses a single, cleanly-specified confounder by construction. Real observational data typically has multiple, partially-unmeasured confounders and non-linear relationships, which would make the adjustment problem harder than shown here. The matching method used is a simple 1:1 nearest-neighbor match without formal covariate balance diagnostics beyond the average propensity-score distance -- a more thorough analysis would report standardized mean differences pre/post matching and consider alternative estimators (inverse propensity weighting, doubly robust estimation) as a robustness check.

---

## 8. Guardrail Trade-off Analysis

Pulling the primary metric, a business-facing revenue translation, and the sustainability guardrail together:

| Metric | Difference (treatment - control) | Significant? |
|---|---|---|
| Conversion rate | -0.0245 | Yes |
| Expected revenue/user ($) | -$1.24 | No |
| Guardrail CO2e/user (kg) | -0.710 | No |
| Green-alt uptake (among converters) | +0.0882 | Yes |

Notably, the revenue translation of the conversion drop ($52.17 to $50.93 per user, t = -1.482, p = 0.1385) is *not* statistically significant even though the conversion-rate difference itself is -- revenue carries additional noise from the cart-value distribution on top of the conversion signal, so it takes more data to detect reliably at the same confidence level.

Because the guardrail CO2e result isn't statistically significant, a precise "$ of revenue per kg of CO2e" cost-effectiveness ratio isn't reported -- dividing by a noisy, statistically-indistinguishable-from-zero number would manufacture false precision. The honest conclusion is that this experiment reliably detected a *behavioral* shift (more green choices among buyers) and a reliable conversion cost, but did not detect a reliable net change in total footprint once both effects are combined.

**What a decision-maker would weigh:**
- The nudge reliably costs conversions (and, directionally, revenue, though that specific estimate isn't statistically significant at this sample size).
- It reliably increases green-alternative selection among people who do convert.
- It does not reliably reduce total footprint once the conversion loss is priced in -- the conversion-loss and greener-choice effects appear to roughly offset each other in this simulation.
- A recommendation would hinge on whether the business values the behavioral signal (more people choosing green when given the option) on its own, even without a proven aggregate footprint reduction, and whether a redesigned, lower-friction nudge could preserve the green-choice lift while recovering some of the conversion cost.

---

## 9. Recommendations

1. **Don't ship the current nudge design as-is on footprint-reduction grounds alone** -- the guardrail metric doesn't support a footprint-reduction claim at this sample size, even though the behavioral shift is real and statistically solid.
2. **Consider iterating on nudge design** to reduce the conversion-cost side (e.g., less intrusive placement, or making the lower-carbon alternative more prominent/easier to select) while preserving the green-alt lift, then re-test.
3. **If footprint reduction is the primary success criterion**, a follow-up experiment needs a substantially larger sample (or a longer run) to have a chance of detecting a guardrail-level effect with confidence, especially if more than two arms are being tested at once.
4. **Fix the electronics emission-factor gap** before treating category-level footprint comparisons as fully trustworthy -- one of five categories currently relies on a placeholder rather than a sourced Climatiq factor.
5. **If this were a real product decision**, prefer the RCT-based estimate over any observational shortcut -- Section 7 demonstrates concretely how badly a naive observational estimate can mislead, even reversing the sign of the true effect.

---

## 10. Limitations

- All effect sizes (conversion friction, green-alt lift, per-variant effects, confounding strength) were author-set assumptions for the purpose of demonstrating the methodology, not measured from real user behavior. They're documented as constants directly in each script.
- One of five product categories (electronics) falls back to a placeholder emission factor rather than a real, sourced one.
- The observational/causal-inference comparison (Section 7) uses a single, cleanly-specified confounder; real-world confounding is typically messier and only partially measurable.
- The multi-variant test (Section 6) is underpowered by design, illustrating a power/breadth trade-off rather than a definitive verdict on any specific variant.
- Propensity-score matching used a simple 1:1 nearest-neighbor approach without formal balance diagnostics beyond match distance.

---

## Appendix: How to Reproduce

All scripts live in `src/`; generated data lands in `data/` (gitignored). Run from inside `src/` in this order (later scripts depend on the CSVs earlier ones produce):

1. `fetch_reference_factors.py` -> `data/reference_factors.csv`
2. `simulate_experiment.py` -> `data/experiment_data.csv`
3. `power_analysis.py` (standalone)
4. `ab_test_analysis.py`
5. `cuped_analysis.py`
6. `sequential_testing.py`
7. `multi_variant_experiment.py` -> `data/multi_variant_data.csv`
8. `fdr_correction.py`
9. `simulate_observational.py` -> `data/observational_data.csv`
10. `causal_comparison.py`
11. `guardrail_analysis.py` (imports helper functions from `ab_test_analysis.py`)

See the [README](../README.md) for environment setup instructions.
