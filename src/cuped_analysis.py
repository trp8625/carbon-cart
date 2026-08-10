"""
CUPED (Controlled-experiment Using Pre-Experiment Data) applied to the
conversion metric, using historical_avg_spend as the pre-experiment
covariate.

Reference: Deng, Xu, Kohavi & Walker, "Improving the Sensitivity of Online
Controlled Experiments by Utilizing Pre-Experiment Data" (WSDM 2013). I'm
confident this is the correct/standard citation for CUPED, but if you're
citing it in a write-up, verify the exact venue/year against the paper
itself rather than trusting this comment alone.

Method:
  theta = Cov(Y, X) / Var(X)          (estimated pooled across both arms)
  Y_cuped = Y - theta * (X - mean(X))

Y_cuped has the same expectation as Y but lower variance whenever X is
correlated with Y -- which is why it needs a t-test on means rather than
a proportions z-test afterward, even though the original Y was binary.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALPHA = 0.05


def compute_cuped(y, x):
    """Returns (y_cuped, theta)."""
    x_mean = x.mean()
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_cuped = y - theta * (x - x_mean)
    return y_cuped, theta


def mean_test(label, treat_values, ctrl_values):
    mean_treat, mean_ctrl = treat_values.mean(), ctrl_values.mean()
    cm = CompareMeans(DescrStatsW(treat_values), DescrStatsW(ctrl_values))
    tstat, pvalue, _ = cm.ttest_ind(usevar="unequal")
    ci_low, ci_high = cm.tconfint_diff(usevar="unequal")

    sig = "SIGNIFICANT" if pvalue < ALPHA else "not significant"
    print(f"  Control mean:   {mean_ctrl:.5f}")
    print(f"  Treatment mean: {mean_treat:.5f}")
    print(f"  Difference: {mean_treat - mean_ctrl:+.5f}, 95% CI [{ci_low:+.5f}, {ci_high:+.5f}]")
    print(f"  t = {tstat:.3f}, p = {pvalue:.4f}  -> {sig} at alpha={ALPHA}")

    return pvalue, mean_treat - mean_ctrl


def main():
    df = pd.read_csv(DATA_DIR / "experiment_data.csv")
    y = df["converted"].to_numpy(dtype=float)
    x = df["historical_avg_spend"].to_numpy(dtype=float)
    treatment = df["treatment"].to_numpy()

    y_cuped, theta = compute_cuped(y, x)
    df["converted_cuped"] = y_cuped

    print(f"theta (regression coefficient of Y on X): {theta:.6f}")
    correlation = np.corrcoef(y, x)[0, 1]
    print(f"Correlation between conversion and historical_avg_spend: {correlation:.4f}")

    var_raw = np.var(y, ddof=1)
    var_cuped = np.var(y_cuped, ddof=1)
    reduction_pct = (1 - var_cuped / var_raw) * 100
    print(f"\nVariance without CUPED: {var_raw:.6f}")
    print(f"Variance with CUPED:    {var_cuped:.6f}")
    print(f"Variance reduction: {reduction_pct:.2f}%")
    if reduction_pct < 1:
        print("(Reduction is minimal -- this means historical_avg_spend isn't strongly "
              "correlated with conversion in this dataset. CUPED's benefit is entirely "
              "driven by that correlation; a weak covariate gives a weak (or negative) "
              "reduction. This is expected/correct behavior, not a bug.)")

    treat_mask = treatment == 1
    ctrl_mask = treatment == 0

    print("\n--- WITHOUT CUPED (raw conversion, treatment vs control) ---")
    p_raw, diff_raw = mean_test("raw", y[treat_mask], y[ctrl_mask])

    print("\n--- WITH CUPED (adjusted conversion, treatment vs control) ---")
    p_cuped, diff_cuped = mean_test("cuped", y_cuped[treat_mask], y_cuped[ctrl_mask])

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Estimated effect without CUPED: {diff_raw:+.5f} (p={p_raw:.4f})")
    print(f"Estimated effect with CUPED:    {diff_cuped:+.5f} (p={p_cuped:.4f})")
    print("Note: CUPED should not change the estimated effect size much (both are "
          "unbiased estimators of the same treatment effect) -- what it should change "
          "is the p-value/CI width, by reducing noise unrelated to treatment.")


if __name__ == "__main__":
    main()
