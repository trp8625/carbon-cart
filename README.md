# Carbon Cart: An A/B Testing Case Study

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Does showing shoppers a real-time carbon footprint estimate at checkout change their behavior — and at what cost? This project runs a full, realistic A/B testing pipeline on a simulated e-commerce checkout flow, using **real carbon emission data from the [Climatiq API](https://www.climatiq.io/)** rather than made-up numbers, and goes well beyond a basic significance test.

**[Read the full write-up →](docs/REPORT.md)**

## Key Findings

- The nudge produced a statistically significant **-2.45pp drop in conversion** (p = 0.0006) and a significant **+8.82pp increase** in shoppers choosing a lower-carbon alternative (p < 0.0001).
- The **guardrail metric — total footprint per user — did not move significantly** (p = 0.39): the conversion loss and the greener choices roughly offset each other.
- A parallel causal-inference exercise showed that estimating this same effect from **biased observational data flips the sign of the result entirely** — a concrete demonstration of why randomization matters.

## Methodology

The pipeline covers:

- **Power analysis** to size the experiment before collecting any data (`statsmodels`)
- **CUPED** variance reduction using a pre-experiment covariate
- **Sequential testing correction** — Monte Carlo–calibrated to show naive "peeking" inflates false positives from 5% to over 20%, then corrects for it
- **Multi-variant testing with Benjamini-Hochberg FDR correction** across simultaneous nudge designs
- **Causal inference comparison** — propensity score matching (`scikit-learn`) vs. a naive observational estimate, benchmarked against the randomized ground truth
- **Guardrail trade-off analysis** — primary metric, revenue translation, and sustainability metric analyzed together, not in isolation

## Data Source

Category-level carbon intensities (kg CO2e per $1 spent) are pulled live from Climatiq's emission factor database — real, sourced, published data, not invented figures. See [`docs/REPORT.md`](docs/REPORT.md#21-real-emission-factors-climatiq-api) for the exact factors used and a documented gap (one category falls back to a placeholder where no match was found).

## Tech Stack

Python · pandas · NumPy · statsmodels · scikit-learn · SciPy · Climatiq REST API

## Repo Structure

```
.
├── README.md
├── docs/
│   └── REPORT.md              # full technical write-up
├── requirements.txt
├── .env.example                # copy to .env and add your Climatiq API key
└── src/
    ├── climatiq_client.py      # Climatiq API wrapper (search + estimate)
    ├── fetch_reference_factors.py
    ├── power_analysis.py
    ├── simulate_experiment.py
    ├── ab_test_analysis.py
    ├── cuped_analysis.py
    ├── sequential_testing.py
    ├── multi_variant_experiment.py
    ├── fdr_correction.py
    ├── simulate_observational.py
    ├── causal_comparison.py
    └── guardrail_analysis.py
```

Generated data lands in `data/` (gitignored — regenerate it by running the scripts below).

## Getting Started

```bash
git clone https://github.com/<your-username>/carbon-nudge-ab-test.git
cd carbon-nudge-ab-test
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then add your free Climatiq API key
```

Get a free API key at [app.climatiq.io](https://app.climatiq.io) (no credit card required at time of writing).

Then run the pipeline from `src/`, in order:

```bash
cd src
python fetch_reference_factors.py
python simulate_experiment.py
python power_analysis.py
python ab_test_analysis.py
python cuped_analysis.py
python sequential_testing.py
python multi_variant_experiment.py
python fdr_correction.py
python simulate_observational.py
python causal_comparison.py
python guardrail_analysis.py
```

Full explanation of each step, all results, and the reasoning behind every design choice: **[`docs/REPORT.md`](docs/REPORT.md)**.

## License

MIT — see [LICENSE](LICENSE).
