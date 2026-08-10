"""
Multiple-testing correction across the three variants in
data/multi_variant_data.csv (each compared against control), using
Benjamini-Hochberg to control the false discovery rate (FDR).

Why this matters: running 3 separate variant-vs-control tests at
alpha=0.05 each means roughly a 1 - 0.95^3 = 14% chance of at least one
false "significant" result even if NONE of the variants actually do
anything -- the same multiple-comparisons problem as sequential peeking,
but across parallel variants instead of across time.

Uses statsmodels.stats.multitest.multipletests(method="fdr_bh"), verified
against the installed statsmodels version before use.
"""

from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALPHA = 0.05


def variant_vs_control_pvalue(df, variant_name, control_name="control", metric="converted"):
    variant_rows = df[df.variant == variant_name]
    control_rows = df[df.variant == control_name]
    count = [variant_rows[metric].sum(), control_rows[metric].sum()]
    nobs = [len(variant_rows), len(control_rows)]
    zstat, pvalue = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")
    diff = variant_rows[metric].mean() - control_rows[metric].mean()
    return zstat, pvalue, diff


def main():
    df = pd.read_csv(DATA_DIR / "multi_variant_data.csv")
    variants = [v for v in df["variant"].unique() if v != "control"]

    results = []
    for v in variants:
        zstat, pvalue, diff = variant_vs_control_pvalue(df, v, metric="converted")
        results.append({"variant": v, "zstat": zstat, "pvalue": pvalue, "diff": diff})

    results_df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)

    # Naive: each test judged against alpha=0.05 independently
    results_df["naive_significant"] = results_df["pvalue"] < ALPHA

    # Corrected: Benjamini-Hochberg FDR correction across all tests together
    reject, pvals_corrected, _, _ = multipletests(results_df["pvalue"], alpha=ALPHA, method="fdr_bh")
    results_df["bh_pvalue_adjusted"] = pvals_corrected
    results_df["bh_significant"] = reject

    print(f"Conversion rate: each variant vs. control, {len(variants)} comparisons\n")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    flipped = results_df[results_df["naive_significant"] & ~results_df["bh_significant"]]
    if len(flipped) > 0:
        print(f"\n{len(flipped)} variant(s) looked significant naively but NOT after "
              f"BH correction -- these would have been false discoveries if you'd "
              f"judged each test independently:")
        print(flipped[["variant", "pvalue", "bh_pvalue_adjusted"]].to_string(index=False))
    else:
        print("\nNo naive-vs-corrected disagreements in this run -- BH correction didn't "
              "flip any conclusions here, but rerun with different assumed effect sizes "
              "in multi_variant_experiment.py to see cases where it does.")


if __name__ == "__main__":
    main()
