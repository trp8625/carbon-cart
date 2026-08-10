"""
Power analysis for the core A/B test (conversion rate).

Run this BEFORE generating any simulated data -- it tells you how many
users per group you need to reliably detect a given effect size.

Uses statsmodels.stats.power.NormalIndPower and
statsmodels.stats.proportion.proportion_effectsize (two-proportion z-test
via Cohen's h effect size), verified against the installed statsmodels
version before use. If you're on a very different version, cross-check
against the docs:
https://www.statsmodels.org/stable/generated/statsmodels.stats.power.NormalIndPower.html
"""

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# ---- ASSUMPTIONS (adjust to your scenario -- these are illustrative, not researched benchmarks) ----

# Baseline checkout conversion rate. Real checkout-stage conversion varies
# a lot by industry/definition -- verify against a primary source if you
# want to defend this number in a write-up rather than label it assumed.
BASELINE_CONVERSION = 0.70

# Minimum detectable effect: smallest true conversion-rate change (in
# absolute percentage points) you want to reliably detect.
MDE_CANDIDATES = [0.01, 0.02, 0.03, 0.05]

ALPHA = 0.05
POWER = 0.80


def required_sample_size(baseline, mde, alpha=ALPHA, power=POWER):
    p1 = baseline
    p2 = baseline - mde
    effect_size = proportion_effectsize(p1, p2)  # Cohen's h
    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return n_per_group


if __name__ == "__main__":
    print(f"Baseline conversion assumed: {BASELINE_CONVERSION:.0%}")
    print(f"Alpha: {ALPHA}, Power: {POWER}\n")
    print(f"{'MDE (pp)':>10} | {'n per group':>12} | {'total n':>10}")
    for mde in MDE_CANDIDATES:
        n = required_sample_size(BASELINE_CONVERSION, mde)
        print(f"{mde*100:>9.1f}% | {n:>12.0f} | {n*2:>10.0f}")
