"""
Generates a synthetic OBSERVATIONAL dataset for the causal-inference
comparison, where "seeing the carbon nudge" is NOT randomly assigned --
unlike data/experiment_data.csv from the real A/B test.

Story: imagine the carbon-estimate feature was rolled out as an opt-in
beta, and more eco-conscious users self-selected into seeing it. That
means eco-consciousness is a CONFOUNDER: it affects both (a) who saw the
nudge and (b) conversion behavior directly, regardless of the nudge
itself. A naive treated-vs-untreated comparison on this data will be
biased; causal_comparison.py (next script) shows how much, and whether
propensity score matching can recover the true effect.

TRUE_EFFECT is known here because we're simulating it -- that's what lets
us grade the observational estimators against ground truth. In a real
observational study you would never know this.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N_USERS = 16800
RANDOM_SEED = 7

CART_VALUE_MEAN_LOG = np.log(60)
CART_VALUE_SIGMA = 0.6
BASELINE_CONVERSION = 0.70

# The TRUE causal effect of seeing the nudge on conversion probability.
# Matches the magnitude assumed in the randomized simulation
# (TREATMENT_CONVERSION_EFFECT in simulate_experiment.py) so the two are
# comparable.
TRUE_EFFECT = -0.02

# --- Confounding structure (assumptions) ---
# eco_score: latent "eco-consciousness" trait, standardized (mean 0, sd 1).
# It affects BOTH treatment assignment and the outcome directly, which is
# exactly what makes it a confounder.
ECO_SCORE_ON_TREATMENT_LOGIT = 1.0   # more eco-conscious users are much more likely to have opted in
ECO_SCORE_ON_CONVERSION = 0.05       # eco-conscious users convert somewhat more, independent of the nudge
SPEND_CONVERSION_EFFECT = 0.03       # same mechanism as the randomized simulation


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n_users=N_USERS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    user_id = np.arange(n_users)
    eco_score = rng.normal(0, 1, size=n_users)

    cart_value = rng.lognormal(mean=CART_VALUE_MEAN_LOG, sigma=CART_VALUE_SIGMA, size=n_users).round(2)
    historical_avg_spend = (cart_value * rng.uniform(0.8, 1.2, size=n_users)).round(2)
    spend_z = (historical_avg_spend - historical_avg_spend.mean()) / historical_avg_spend.std()

    # Treatment assignment DEPENDS on eco_score -- this is the confounding,
    # not random assignment like the real A/B test.
    treat_prob = sigmoid(ECO_SCORE_ON_TREATMENT_LOGIT * eco_score)
    saw_nudge = rng.binomial(1, treat_prob)

    conversion_prob = np.clip(
        BASELINE_CONVERSION
        + TRUE_EFFECT * saw_nudge
        + ECO_SCORE_ON_CONVERSION * eco_score
        + SPEND_CONVERSION_EFFECT * spend_z,
        0.01, 0.99,
    )
    converted = rng.binomial(1, conversion_prob)

    return pd.DataFrame({
        "user_id": user_id,
        "eco_score": eco_score,
        "cart_value": cart_value,
        "historical_avg_spend": historical_avg_spend,
        "saw_nudge": saw_nudge,
        "converted": converted,
    })


if __name__ == "__main__":
    df = simulate()
    out_path = DATA_DIR / "observational_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print(f"True causal effect (known, by construction): {TRUE_EFFECT:+.4f}")
    print(f"\nShare who saw the nudge: {df['saw_nudge'].mean():.3f}")
    print(f"Mean eco_score among nudge-seers:    {df[df.saw_nudge == 1].eco_score.mean():+.3f}")
    print(f"Mean eco_score among non-nudge-seers: {df[df.saw_nudge == 0].eco_score.mean():+.3f}")
    print("(These should differ a lot -- that's the confounding.)")
    print(f"\nNaive conversion, saw nudge:     {df[df.saw_nudge == 1].converted.mean():.4f}")
    print(f"Naive conversion, didn't see it: {df[df.saw_nudge == 0].converted.mean():.4f}")
