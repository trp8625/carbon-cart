"""
Core A/B test analysis on data/experiment_data.csv (produced by simulate_experiment.py).

Three tests, in order of importance:
  1. PRIMARY: conversion rate, treatment vs control (two-proportion z-test)
  2. Green-alternative selection rate among converters (two-proportion z-test)
  3. GUARDRAIL: effective CO2e per user shown checkout (Welch's t-test) --
     this is cart_co2e_kg for converters and 0 for non-converters, since a
     cart that was never purchased has no real-world emissions.

All functions verified against the installed statsmodels/scipy versions
before use:
  - statsmodels.stats.proportion.proportions_ztest
  - statsmodels.stats.proportion.confint_proportions_2indep
  - statsmodels.stats.weightstats.CompareMeans / DescrStatsW
"""

from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALPHA = 0.05


def proportion_test(label, count_treat, n_treat, count_ctrl, n_ctrl):
    rate_treat = count_treat / n_treat
    rate_ctrl = count_ctrl / n_ctrl

    zstat, pvalue = proportions_ztest(
        count=[count_treat, count_ctrl],
        nobs=[n_treat, n_ctrl],
        alternative="two-sided",
    )
    ci_low, ci_high = confint_proportions_2indep(
        count1=count_treat, nobs1=n_treat,
        count2=count_ctrl, nobs2=n_ctrl,
        method="wald", compare="diff",
    )

    sig = "SIGNIFICANT" if pvalue < ALPHA else "not significant"
    print(f"\n--- {label} ---")
    print(f"Control:   {rate_ctrl:.4f}  (n={n_ctrl})")
    print(f"Treatment: {rate_treat:.4f}  (n={n_treat})")
    print(f"Difference (treatment - control): {rate_treat - rate_ctrl:+.4f}")
    print(f"95% CI on difference: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"z = {zstat:.3f}, p = {pvalue:.4f}  -> {sig} at alpha={ALPHA}")

    return {"metric": label, "control": rate_ctrl, "treatment": rate_treat,
            "diff": rate_treat - rate_ctrl, "ci_low": ci_low, "ci_high": ci_high,
            "pvalue": pvalue, "significant": pvalue < ALPHA}


def mean_test(label, treat_values, ctrl_values):
    mean_treat, mean_ctrl = treat_values.mean(), ctrl_values.mean()

    cm = CompareMeans(DescrStatsW(treat_values), DescrStatsW(ctrl_values))
    tstat, pvalue, _ = cm.ttest_ind(usevar="unequal")
    ci_low, ci_high = cm.tconfint_diff(usevar="unequal")

    sig = "SIGNIFICANT" if pvalue < ALPHA else "not significant"
    print(f"\n--- {label} ---")
    print(f"Control mean:   {mean_ctrl:.4f}")
    print(f"Treatment mean: {mean_treat:.4f}")
    print(f"Difference (treatment - control): {mean_treat - mean_ctrl:+.4f}")
    print(f"95% CI on difference: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"t = {tstat:.3f}, p = {pvalue:.4f}  -> {sig} at alpha={ALPHA}")

    return {"metric": label, "control": mean_ctrl, "treatment": mean_treat,
            "diff": mean_treat - mean_ctrl, "ci_low": ci_low, "ci_high": ci_high,
            "pvalue": pvalue, "significant": pvalue < ALPHA}


def main():
    df = pd.read_csv(DATA_DIR / "experiment_data.csv")
    treat = df[df.treatment == 1]
    ctrl = df[df.treatment == 0]

    results = []
    results.append(proportion_test(
        "Conversion rate",
        treat.converted.sum(), len(treat),
        ctrl.converted.sum(), len(ctrl),
    ))

    treat_conv = treat[treat.converted == 1]
    ctrl_conv = ctrl[ctrl.converted == 1]
    results.append(proportion_test(
        "Green-alternative selection rate (among converters)",
        treat_conv.chose_green_alt.sum(), len(treat_conv),
        ctrl_conv.chose_green_alt.sum(), len(ctrl_conv),
    ))

    df["effective_co2e"] = df["cart_co2e_kg"] * df["converted"]
    treat_all = df[df.treatment == 1]
    ctrl_all = df[df.treatment == 0]
    results.append(mean_test(
        "Effective cart CO2e per user shown checkout (kg)",
        treat_all.effective_co2e, ctrl_all.effective_co2e,
    ))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        flag = "***" if r["significant"] else "   "
        print(f"{flag} {r['metric']}: diff={r['diff']:+.4f}, p={r['pvalue']:.4f}")


if __name__ == "__main__":
    main()
