"""
Sequential testing / peeking correction, applied to the conversion metric.

Demonstrates two things:
  1. THE PROBLEM: repeatedly checking a fixed-alpha=0.05 test as data
     accumulates inflates the true false-positive rate well above 5%,
     because "stop as soon as any interim look is significant" is
     implicitly many correlated tests, not one.
  2. A FIX: rather than trusting a closed-form Pocock/O'Brien-Fleming
     boundary formula from memory (which I'd rather not risk getting
     subtly wrong), this calibrates a single stricter |z| threshold by
     Monte Carlo simulation: simulate many NULL experiments (no true
     treatment effect) with the same number of looks and look-sizes as
     the real data, and find the cutoff that keeps the any-look
     false-positive rate at alpha. This is slower than a formula but is
     fully verifiable by running the simulation.

Everything here is testable/reproducible; if you want to compare against
a textbook Pocock or O'Brien-Fleming boundary, verify the formula against
a primary source (e.g. Jennison & Turnbull's "Group Sequential Methods")
before using it -- I'm not citing one here from memory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ALPHA = 0.05
N_LOOKS = 10        # number of interim peeks (e.g. simulating 10 "days" of accumulating data)
N_NULL_SIMS = 3000  # Monte Carlo reps used to calibrate the corrected threshold and estimate the naive FPR


def cumulative_zstats(converted, treatment, n_looks=N_LOOKS):
    """
    Splits an (already-collected) dataset into n_looks equal-sized
    cumulative chunks -- as if data arrived in that many batches, e.g.
    days -- and computes the conversion z-statistic (treatment vs
    control) at each cumulative look. Returns an array of z-statistics,
    one per look.
    """
    n = len(converted)
    look_sizes = np.linspace(n / n_looks, n, n_looks).astype(int)
    zstats = np.empty(n_looks)
    for i, size in enumerate(look_sizes):
        c = converted[:size]
        t = treatment[:size]
        treat_mask = t == 1
        ctrl_mask = t == 0
        count = [c[treat_mask].sum(), c[ctrl_mask].sum()]
        nobs = [treat_mask.sum(), ctrl_mask.sum()]
        zstat, _ = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")
        zstats[i] = zstat
    return zstats


def simulate_null_max_abs_z(n_total, n_looks, baseline_rate, n_sims, seed):
    """For n_sims null experiments (no true effect), returns the max |z| across all looks, per simulation."""
    rng = np.random.default_rng(seed)
    max_abs_z = np.empty(n_sims)
    for i in range(n_sims):
        converted = rng.binomial(1, baseline_rate, size=n_total)
        treatment = rng.integers(0, 2, size=n_total)
        zstats = cumulative_zstats(converted, treatment, n_looks)
        max_abs_z[i] = np.max(np.abs(zstats))
    return max_abs_z


def main():
    df = pd.read_csv(DATA_DIR / "experiment_data.csv")
    converted = df["converted"].to_numpy()
    treatment = df["treatment"].to_numpy()
    n_total = len(df)
    baseline_rate = converted[treatment == 0].mean()  # used only to calibrate the null simulation

    z_crit_naive = norm.ppf(1 - ALPHA / 2)  # standard two-sided z critical value, e.g. ~1.96 at alpha=0.05

    # --- Step 1: what would naive daily peeking have concluded on the real data? ---
    real_zstats = cumulative_zstats(converted, treatment, N_LOOKS)
    print(f"Z-statistics at each of {N_LOOKS} cumulative looks (real data):")
    for i, z in enumerate(real_zstats, 1):
        flag = " <- naive stop (|z| > 1.96)" if abs(z) > z_crit_naive else ""
        print(f"  Look {i:2d}: z = {z:+.3f}{flag}")

    naive_stop_look = next((i for i, z in enumerate(real_zstats, 1) if abs(z) > z_crit_naive), None)
    if naive_stop_look:
        print(f"\nNaive rule would have stopped at look {naive_stop_look} and declared significance.")
    else:
        print("\nNaive rule never crossed the uncorrected threshold at any look.")

    # --- Step 2: how inflated is the false-positive rate under naive peeking? ---
    print(f"\nCalibrating via {N_NULL_SIMS} simulated null experiments "
          f"(baseline rate={baseline_rate:.3f}, {N_LOOKS} looks each)...")
    null_max_abs_z = simulate_null_max_abs_z(n_total, N_LOOKS, baseline_rate, N_NULL_SIMS, seed=1)

    naive_fpr = np.mean(null_max_abs_z > z_crit_naive)
    print(f"\nNaive false-positive rate (stop at first |z| > 1.96, {N_LOOKS} looks): {naive_fpr:.3f}")
    print(f"  (target/nominal alpha was {ALPHA}; naive peeking inflates this if > {ALPHA})")

    # --- Step 3: corrected threshold that controls FPR at alpha across all looks ---
    z_crit_corrected = np.quantile(null_max_abs_z, 1 - ALPHA)
    print(f"\nMonte-Carlo-calibrated corrected threshold: |z| > {z_crit_corrected:.3f} "
          f"(vs. uncorrected {z_crit_naive:.3f})")

    corrected_stop_look = next((i for i, z in enumerate(real_zstats, 1) if abs(z) > z_crit_corrected), None)
    print("\n--- CORRECTED conclusion on the real data ---")
    if corrected_stop_look:
        print(f"Corrected rule stops at look {corrected_stop_look} and declares significance "
              f"(z = {real_zstats[corrected_stop_look - 1]:+.3f}).")
    else:
        print("Corrected rule never crosses the calibrated threshold at any look -- "
              "no significant result under valid sequential testing, even if a naive "
              "peek looked significant along the way.")


if __name__ == "__main__":
    main()
