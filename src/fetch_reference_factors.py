"""
One-time script: populates data/reference_factors.csv with REAL Climatiq
emission factors (kg CO2e per $1 spent) for each product category used in
simulate_experiment.py.

Run this once, with your real API key in .env (at the project root),
before running simulate_experiment.py -- that way the simulation uses
real emission data instead of the placeholder fallback numbers.
"""

import csv
from pathlib import Path

from climatiq_client import search_emission_factors, estimate_emissions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Free-text search query per category, targeting Climatiq's spend-based
# (unit_type "Money") emission factors. These queries are a starting
# point, not verified "correct" matches -- the script prints the exact
# factor it picked for each category so you can sanity-check it (or swap
# in a better-matching activity_id by hand) before trusting the results.
CATEGORY_QUERIES = {
    "electronics": "computer and electronic product manufacturing",
    "apparel": "apparel manufacturing",
    "home_goods": "furniture manufacturing",
    "food_beverage": "food and beverage stores",
    "books_media": "publishing industries",
}


def find_money_based_factor(query):
    results = search_emission_factors(query, unit_type="Money", results_per_page=5)
    return results[0] if results else None  # best fuzzy match


def main():
    rows = []
    for category, query in CATEGORY_QUERIES.items():
        match = find_money_based_factor(query)
        if match is None:
            print(f"[WARN] No money-based factor found for '{category}' (query: {query!r}). Skipping -- "
                  f"simulate_experiment.py will fall back to a placeholder for this category.")
            continue

        estimate = estimate_emissions(
            activity_id=match["activity_id"],
            parameters={"money": 1, "money_unit": "usd"},
        )
        kg_per_dollar = estimate["co2e"]
        print(f"{category}: {match['name']} ({match['activity_id']}, region={match.get('region', 'n/a')}) "
              f"-> {kg_per_dollar} kgCO2e per $1")

        rows.append({
            "category": category,
            "activity_id": match["activity_id"],
            "factor_name": match["name"],
            "region": match.get("region", ""),
            "kg_co2e_per_dollar": kg_per_dollar,
        })

    out_path = DATA_DIR / "reference_factors.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "activity_id", "factor_name", "region", "kg_co2e_per_dollar"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} category factors to {out_path}")
    if len(rows) < len(CATEGORY_QUERIES):
        print("Some categories are missing a real factor -- review the [WARN] lines above and consider "
              "refining CATEGORY_QUERIES or picking an activity_id manually via the Data Explorer "
              "(https://climatiq.io/data).")


if __name__ == "__main__":
    main()
