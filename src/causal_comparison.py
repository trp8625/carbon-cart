"""
Compares three ways of estimating the same treatment effect on
conversion:

  1. NAIVE observational: raw diff-in-means, saw_nudge vs didn't,
     on data/observational_data.csv. Biased, because saw_nudge wasn't
     randomly assigned (eco_score confounds both treatment and outcome).
  2. PROPENSITY-SCORE-MATCHED observational: estimate each user's
     probability of having seen the nudge (the propensity score) from
     eco_score and historical_avg_spend, then match each nudge-seer to
     their nearest untreated look-alike and compare outcomes within
     matched pairs. This should recover something much closer to the
     true effect, by comparing users who were equally *likely* to see
     the nudge but happened to differ in whether they did.
  3. RANDOMIZED (RCT): the actual A/B test result from
     data/experiment_data.csv, where treatment WAS randomly assigned --
     the unbiased ground truth, used here as a benchmark, not because
     we'd normally have this available in a real observational study.

Propensity score matching implemented directly with
sklearn.linear_model.LogisticRegression (propensity model) and
sklearn.neighbors.NearestNeighbors (1:1 nearest-neighbor matching on the
propensity score) -- both verified against the installed scikit-learn
version before use, rather than relying on a specialized causal-inference
library's API from memory.

This is a standard, well-established design (see e.g. Rosenbaum & Rubin,
1983, on propensity score matching) but if you want to cite the technique
in a write-up, verify the reference yourself rather than trusting this
comment alone.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def naive_estimate(df):
    treated = df[df.saw_nudge == 1].converted
    control = df[df.saw_nudge == 0].converted
    return treated.mean() - control.mean()


def propensity_score_match(df, covariates=("eco_score", "historical_avg_spend")):
    X = df[list(covariates)].to_numpy()
    # Standardize covariates so both features contribute comparably to the propensity model.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = df["saw_nudge"].to_numpy()

    model = LogisticRegression()
    model.fit(X, y)
    propensity = model.predict_proba(X)[:, 1]
    df = df.assign(propensity=propensity)

    treated = df[df.saw_nudge == 1]
    control = df[df.saw_nudge == 0]

    # 1:1 nearest-neighbor matching: for each treated user, find the
    # untreated user with the closest propensity score.
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity"]].to_numpy())
    distances, indices = nn.kneighbors(treated[["propensity"]].to_numpy())

    matched_control = control.iloc[indices.flatten()]
    matched_diff = treated["converted"].to_numpy() - matched_control["converted"].to_numpy()

    ate = matched_diff.mean()
    se = matched_diff.std(ddof=1) / np.sqrt(len(matched_diff))
    ci_low, ci_high = ate - 1.96 * se, ate + 1.96 * se
    mean_match_distance = distances.mean()

    return ate, ci_low, ci_high, mean_match_distance


def rct_estimate(path=None):
    """The real randomized A/B test result, as a ground-truth benchmark."""
    path = path or (DATA_DIR / "experiment_data.csv")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    treated = df[df.treatment == 1].converted
    control = df[df.treatment == 0].converted
    return treated.mean() - control.mean()


def main():
    obs = pd.read_csv(DATA_DIR / "observational_data.csv")
    true_effect = -0.02  # matches TRUE_EFFECT in simulate_observational.py

    naive = naive_estimate(obs)
    psm_ate, psm_ci_low, psm_ci_high, match_quality = propensity_score_match(obs)
    rct = rct_estimate()

    print("=" * 70)
    print("CAUSAL ESTIMATE COMPARISON (effect of nudge on conversion)")
    print("=" * 70)
    print(f"{'Method':<32} {'Estimate':>10} {'Bias vs true':>14}")
    print(f"{'True effect (known, simulated)':<32} {true_effect:>+10.4f} {'--':>14}")
    print(f"{'Naive observational':<32} {naive:>+10.4f} {naive - true_effect:>+14.4f}")
    print(f"{'Propensity-matched observational':<32} {psm_ate:>+10.4f} {psm_ate - true_effect:>+14.4f}")
    if rct is not None:
        print(f"{'Randomized (RCT, actual A/B test)':<32} {rct:>+10.4f} {rct - true_effect:>+14.4f}")
    else:
        print("(RCT comparison skipped -- experiment_data.csv not found; run simulate_experiment.py first.)")

    print(f"\nPSM 95% CI on matched effect: [{psm_ci_low:+.4f}, {psm_ci_high:+.4f}]")
    print(f"Mean propensity-score distance between matched pairs: {match_quality:.4f} "
          f"(smaller = better matches; large values would mean matches are unreliable)")

    print("\nInterpretation:")
    print("The naive comparison is biased because eco-conscious users both convert more AND "
          "were more likely to see the nudge -- confounding can even flip the apparent sign "
          "of the effect. Propensity matching, by comparing users with similar odds of "
          "having seen the nudge, should land much closer to the true effect than the naive "
          "estimate does, though matching on an incomplete or mismeasured set of covariates "
          "can still leave residual bias -- unlike the RCT, which needs no such adjustment "
          "because randomization itself removes the confounding by design.")


if __name__ == "__main__":
    main()
