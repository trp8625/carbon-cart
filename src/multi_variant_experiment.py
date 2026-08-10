"""
Generates a synthetic MULTI-VARIANT checkout experiment: control plus
three different carbon-nudge designs, tested simultaneously. This is the
dataset used by fdr_correction.py to demonstrate why testing several
variants at once needs a multiple-testing correction.

Reuses the same cart/category/emission-factor logic as
simulate_experiment.py (single-variant version) -- see that file for the
Climatiq integration details. Loads data/reference_factors.csv if
present, otherwise falls back to placeholder emission numbers.

ASSUMPTIONS -- clearly separated below, illustrative for a portfolio
project rather than measured. Adjust freely.

Variants:
  control        -- no carbon information shown
  plain_estimate -- shows the cart's carbon estimate only
  green_alt      -- estimate + a suggested lower-carbon alternative
  social_proof   -- estimate + "X% of shoppers chose a lower-carbon option"

NOTE ON SAMPLE SIZE: splitting traffic across 4 arms instead of 2 reduces
the effective sample size per comparison. A proper multi-arm test should
re-run power_analysis.py accounting for the number of arms; N_USERS below
is kept the same as the single-variant script for simplicity, not because
it's been verified sufficient for 4 arms specifically.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N_USERS = 16800
CART_VALUE_MEAN_LOG = np.log(60)
CART_VALUE_SIGMA = 0.6
SPEND_CONVERSION_EFFECT = 0.03
RANDOM_SEED = 42

CATEGORIES = ["electronics", "apparel", "home_goods", "food_beverage", "books_media"]
CATEGORY_WEIGHTS = [0.25, 0.30, 0.20, 0.15, 0.10]

FALLBACK_FACTORS = {
    "electronics": 0.45, "apparel": 0.30, "home_goods": 0.25,
    "food_beverage": 0.55, "books_media": 0.10,
}

VARIANTS = ["control", "plain_estimate", "green_alt", "social_proof"]
VARIANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

BASELINE_CONVERSION = 0.70
BASELINE_GREEN_ALT_RATE = 0.15

# Per-variant effects relative to control (assumptions, not measured):
CONVERSION_EFFECT = {
    "control": 0.0,
    "plain_estimate": -0.02,   # plain estimate: some friction, no easy alternative offered
    "green_alt": -0.015,       # slightly less friction: offering an easy swap softens the hit
    "social_proof": -0.01,     # social proof framing softens friction further
}
GREEN_ALT_EFFECT = {
    "control": 0.0,
    "plain_estimate": 0.05,    # info alone nudges some people
    "green_alt": 0.12,         # making the swap one click away helps a lot
    "social_proof": 0.08,      # social framing helps, less than a direct one-click swap
}


def load_reference_factors(path=None):
    path = path or (DATA_DIR / "reference_factors.csv")
    try:
        df = pd.read_csv(path)
        factors = dict(zip(df["category"], df["kg_co2e_per_dollar"]))
        missing = [c for c in CATEGORIES if c not in factors]
        for c in missing:
            factors[c] = FALLBACK_FACTORS[c]
        if missing:
            print(f"[WARN] missing real factors for {missing}, using fallback values for those.")
        return factors
    except FileNotFoundError:
        print("data/reference_factors.csv not found -- using placeholder fallback factors for all categories.")
        return FALLBACK_FACTORS


def simulate(n_users=N_USERS, reference_factors=None, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    factors = reference_factors or FALLBACK_FACTORS

    user_id = np.arange(n_users)
    variant = rng.choice(VARIANTS, size=n_users, p=VARIANT_WEIGHTS)
    category = rng.choice(CATEGORIES, size=n_users, p=CATEGORY_WEIGHTS)
    cart_value = rng.lognormal(mean=CART_VALUE_MEAN_LOG, sigma=CART_VALUE_SIGMA, size=n_users).round(2)
    historical_avg_spend = (cart_value * rng.uniform(0.8, 1.2, size=n_users)).round(2)

    factor_per_dollar = np.array([factors[c] for c in category])
    cart_co2e_kg = (cart_value * factor_per_dollar).round(3)

    spend_z = (historical_avg_spend - historical_avg_spend.mean()) / historical_avg_spend.std()
    conversion_effect = np.array([CONVERSION_EFFECT[v] for v in variant])
    conversion_prob = np.clip(BASELINE_CONVERSION + conversion_effect + SPEND_CONVERSION_EFFECT * spend_z, 0.01, 0.99)
    converted = rng.binomial(1, conversion_prob)

    green_alt_effect = np.array([GREEN_ALT_EFFECT[v] for v in variant])
    green_alt_prob = np.clip(BASELINE_GREEN_ALT_RATE + green_alt_effect, 0.01, 0.99)
    chose_green_alt = np.where(converted == 1, rng.binomial(1, green_alt_prob), 0)

    return pd.DataFrame({
        "user_id": user_id,
        "variant": variant,
        "category": category,
        "cart_value": cart_value,
        "historical_avg_spend": historical_avg_spend,
        "cart_co2e_kg": cart_co2e_kg,
        "converted": converted,
        "chose_green_alt": chose_green_alt,
    })


if __name__ == "__main__":
    factors = load_reference_factors()
    df = simulate(reference_factors=factors)
    out_path = DATA_DIR / "multi_variant_data.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(df["variant"].value_counts())
    print("\nConversion rate by variant:")
    print(df.groupby("variant")["converted"].mean().round(4))
    print("\nGreen-alt rate among converters, by variant:")
    print(df[df.converted == 1].groupby("variant")["chose_green_alt"].mean().round(4))
