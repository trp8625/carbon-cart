"""
Generates the synthetic checkout-experiment dataset.

Combines REAL Climatiq emission factors (loaded from data/reference_factors.csv,
produced by fetch_reference_factors.py) with simulated users/carts, assigns
treatment/control, and simulates outcomes (conversion, green-alternative
selection).

ASSUMPTIONS -- everything below is a design choice for a portfolio
project, not a measured fact. Adjust freely; just be able to justify your
choices in the write-up. If reference_factors.csv isn't present, or is
missing a category (fetch_reference_factors.py skips categories it
couldn't find a real match for), this falls back to placeholder emission
numbers for whatever is missing so the script still runs end-to-end.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---- ASSUMPTIONS ----
N_USERS = 16800  # ~ n needed for MDE=2pp at 80% power per power_analysis.py; round up in practice
BASELINE_CONVERSION = 0.70
TREATMENT_CONVERSION_EFFECT = -0.02   # nudge causes checkout friction: -2pp (assumption)
BASELINE_GREEN_ALT_RATE = 0.15        # among converters, % choosing a lower-carbon alternative
TREATMENT_GREEN_ALT_EFFECT = 0.10     # nudge raises green-alt selection by +10pp among converters (assumption)
CART_VALUE_MEAN_LOG = np.log(60)      # lognormal cart value centered ~$60
CART_VALUE_SIGMA = 0.6

# Makes historical_avg_spend an ACTUAL predictor of conversion (people with
# higher past spend convert slightly more often), rather than just a
# cosmetic covariate. This is what gives CUPED something real to reduce
# variance with -- a covariate uncorrelated with the outcome would make
# CUPED nearly a no-op. Effect is in units of "prob shift per z-score of
# historical spend" and is an assumption, not a measured relationship.
SPEND_CONVERSION_EFFECT = 0.03

RANDOM_SEED = 42

CATEGORIES = ["electronics", "apparel", "home_goods", "food_beverage", "books_media"]
CATEGORY_WEIGHTS = [0.25, 0.30, 0.20, 0.15, 0.10]  # rough purchase-mix assumption

FALLBACK_FACTORS = {
    "electronics": 0.45, "apparel": 0.30, "home_goods": 0.25,
    "food_beverage": 0.55, "books_media": 0.10,
}


def load_reference_factors(path=None):
    """
    category -> kg CO2e per $1 spent, from real Climatiq data.

    fetch_reference_factors.py can skip a category if it found no
    money-based match, so this file may not cover every category in
    CATEGORIES. Any missing category is backfilled from FALLBACK_FACTORS
    with a printed warning, rather than crashing later in simulate().
    """
    path = path or (DATA_DIR / "reference_factors.csv")
    df = pd.read_csv(path)
    factors = dict(zip(df["category"], df["kg_co2e_per_dollar"]))

    missing = [c for c in CATEGORIES if c not in factors]
    if missing:
        print(f"[WARN] reference_factors.csv is missing real data for: {missing}. "
              f"Using placeholder fallback values for these categories only. "
              f"Check the [WARN] output from fetch_reference_factors.py to see why, "
              f"and consider adjusting CATEGORY_QUERIES there or picking an activity_id "
              f"manually via the Data Explorer (https://climatiq.io/data).")
        for c in missing:
            factors[c] = FALLBACK_FACTORS[c]

    return factors


def simulate(n_users=N_USERS, reference_factors=None, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    user_id = np.arange(n_users)
    treatment = rng.integers(0, 2, size=n_users)  # 0 = control, 1 = treatment
    category = rng.choice(CATEGORIES, size=n_users, p=CATEGORY_WEIGHTS)
    cart_value = rng.lognormal(mean=CART_VALUE_MEAN_LOG, sigma=CART_VALUE_SIGMA, size=n_users).round(2)

    # Historical average spend -- a pre-experiment covariate correlated with
    # cart_value, used both to compute cart_co2e_kg's realism and as the
    # CUPED covariate in cuped_analysis.py.
    historical_avg_spend = (cart_value * rng.uniform(0.8, 1.2, size=n_users)).round(2)

    factors = reference_factors or FALLBACK_FACTORS
    factor_per_dollar = np.array([factors[c] for c in category])
    cart_co2e_kg = (cart_value * factor_per_dollar).round(3)

    # z-score historical spend so SPEND_CONVERSION_EFFECT is on a
    # consistent, interpretable scale regardless of the raw dollar range.
    spend_z = (historical_avg_spend - historical_avg_spend.mean()) / historical_avg_spend.std()

    conversion_prob = np.where(
        treatment == 1,
        BASELINE_CONVERSION + TREATMENT_CONVERSION_EFFECT,
        BASELINE_CONVERSION,
    ) + SPEND_CONVERSION_EFFECT * spend_z
    conversion_prob = np.clip(conversion_prob, 0.01, 0.99)
    converted = rng.binomial(1, conversion_prob)

    green_alt_prob = np.where(
        treatment == 1,
        BASELINE_GREEN_ALT_RATE + TREATMENT_GREEN_ALT_EFFECT,
        BASELINE_GREEN_ALT_RATE,
    )
    chose_green_alt = np.where(converted == 1, rng.binomial(1, green_alt_prob), 0)

    return pd.DataFrame({
        "user_id": user_id,
        "treatment": treatment,
        "category": category,
        "cart_value": cart_value,
        "historical_avg_spend": historical_avg_spend,
        "cart_co2e_kg": cart_co2e_kg,
        "converted": converted,
        "chose_green_alt": chose_green_alt,
    })


if __name__ == "__main__":
    try:
        factors = load_reference_factors()
        print("Loaded reference factors (real Climatiq data, with fallback for any missing categories).")
    except FileNotFoundError:
        factors = None
        print("data/reference_factors.csv not found -- using placeholder fallback factors for all categories.")
        print("Run fetch_reference_factors.py (with your API key) to generate real data, then re-run this script.")

    df = simulate(reference_factors=factors)
    out_path = DATA_DIR / "experiment_data.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(df.head())
    print(f"\nControl conversion:   {df[df.treatment == 0].converted.mean():.3f}")
    print(f"Treatment conversion: {df[df.treatment == 1].converted.mean():.3f}")
    print(f"Control green-alt rate (among converters):   {df[(df.treatment == 0) & (df.converted == 1)].chose_green_alt.mean():.3f}")
    print(f"Treatment green-alt rate (among converters): {df[(df.treatment == 1) & (df.converted == 1)].chose_green_alt.mean():.3f}")
