"""
Thin wrapper around the Climatiq REST API.

Docs verified directly against climatiq.io/docs:
- Auth:     https://www.climatiq.io/docs/api-reference/authentication
- Search:   https://www.climatiq.io/docs/api-reference/search
- Estimate: https://www.climatiq.io/docs/api-reference/estimate

Requires CLIMATIQ_API_KEY to be set in a .env file at the project root
(see .env.example). Get a free key at https://app.climatiq.io (Starter
plan, no credit card required as of this writing -- verify current terms
before relying on this).
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the project root regardless of the current working directory,
# since scripts live in src/ but .env lives one level up.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("CLIMATIQ_API_KEY")
BASE_URL = "https://api.climatiq.io/data/v1"


def _headers():
    if not API_KEY:
        raise RuntimeError(
            "CLIMATIQ_API_KEY is not set. Copy .env.example to .env (at the project "
            "root) and add your key from https://app.climatiq.io"
        )
    return {"Authorization": f"Bearer {API_KEY}"}


def search_emission_factors(query, data_version="^33", region=None, category=None,
                             unit_type=None, results_per_page=10):
    """
    GET /data/v1/search -- find candidate emission factors matching a free-text query.

    Use this first to find the right `activity_id` before calling estimate_emissions().
    Returns the raw `results` list from the API response.
    """
    params = {
        "query": query,
        "data_version": data_version,
        "results_per_page": results_per_page,
    }
    if region:
        params["region"] = region
    if category:
        params["category"] = category
    if unit_type:
        params["unit_type"] = unit_type

    resp = requests.get(f"{BASE_URL}/search", headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()["results"]


def estimate_emissions(activity_id, parameters, data_version="^33", region=None, year=None):
    """
    POST /data/v1/estimate -- calculate CO2e for a single activity.

    activity_id: the Climatiq activity_id, e.g. "metals-type_steel_section"
                 (find these via search_emission_factors()).
    parameters:  dict matching what the chosen emission factor expects,
                 e.g. {"weight": 100, "weight_unit": "kg"}.

    Returns the full estimation response dict (includes co2e, co2e_unit, etc.).
    """
    emission_factor = {"activity_id": activity_id, "data_version": data_version}
    if region:
        emission_factor["region"] = region
    if year:
        emission_factor["year"] = year

    body = {"emission_factor": emission_factor, "parameters": parameters}

    resp = requests.post(f"{BASE_URL}/estimate", headers=_headers(), json=body)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # Smoke test: run this after adding your real key to .env.
    print("Searching for 'light duty truck' emission factors...")
    results = search_emission_factors("light duty truck", results_per_page=3)
    for r in results:
        print(f"  - {r['activity_id']}  ({r['name']}, {r.get('region', 'n/a')})")

    if results:
        first = results[0]
        print(f"\nEstimating emissions using activity_id: {first['activity_id']}")
        print(f"unit_type: {first['unit_type']}, unit: {first['unit']}")
