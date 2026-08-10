"""
Guardrail trade-off analysis: pulls together the primary metric
(conversion), a business-facing translation of it (revenue), and the
sustainability guardrail metric (effective CO2e per user), then frames
the decision a PM would actually have to make.

Reuses the tested proportion_test() / mean_test() helpers from
ab_test_analysis.py rather than re-implementing them (same folder, plain
import works when running this script directly).
"""

from pathlib import Path

import pandas as pd
from ab_test_analysis import proportion_test, mean_test, ALPHA

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    df = pd.read_csv(DATA_DIR / "experiment_data.csv")
    treat = df[df.treatment == 1]
    ctrl = df[df.treatment == 0]

    print("#" * 60)
    print("GUARDRAIL TRADE-OFF ANALYSIS")
    print("#" * 60)

    # 1. Primary metric: conversion (business cost of the nudge)
    conv = proportion_test(
        "Conversion rate (PRIMARY)",
        treat.converted.sum(), len(treat),
        ctrl.converted.sum(), len(ctrl),
    )

    # 2. Revenue translation: expected revenue per user shown checkout
    #    (cart_value if converted, $0 if not) -- makes the conversion cost
    #    concrete in dollar terms rather than just percentage points.
    df["expected_revenue"] = df["cart_value"] * df["converted"]
    treat_all = df[df.treatment == 1]
    ctrl_all = df[df.treatment == 0]
    revenue = mean_test(
        "Expected revenue per user shown checkout ($)",
        treat_all.expected_revenue, ctrl_all.expected_revenue,
    )

    # 3. Guardrail metric: effective CO2e per user shown checkout
    df["effective_co2e"] = df["cart_co2e_kg"] * df["converted"]
    treat_all = df[df.treatment == 1]
    ctrl_all = df[df.treatment == 0]
    co2e = mean_test(
        "Effective cart CO2e per user shown checkout (kg) (GUARDRAIL)",
        treat_all.effective_co2e, ctrl_all.effective_co2e,
    )

    # 4. Secondary positive signal: green-alternative selection among converters
    treat_conv = treat[treat.converted == 1]
    ctrl_conv = ctrl[ctrl.converted == 1]
    green = proportion_test(
        "Green-alternative selection rate among converters (SECONDARY)",
        treat_conv.chose_green_alt.sum(), len(treat_conv),
        ctrl_conv.chose_green_alt.sum(), len(ctrl_conv),
    )

    print("\n" + "#" * 60)
    print("DECISION SUMMARY")
    print("#" * 60)
    print(f"Conversion:        diff={conv['diff']:+.4f}  significant={conv['significant']}")
    print(f"Revenue/user:      diff={revenue['diff']:+.4f}  significant={revenue['significant']}")
    print(f"Guardrail CO2e:    diff={co2e['diff']:+.4f}  significant={co2e['significant']}")
    print(f"Green-alt uptake:  diff={green['diff']:+.4f}  significant={green['significant']}")

    print("\nCost-effectiveness framing:")
    if co2e["significant"]:
        ratio = revenue["diff"] / co2e["diff"]
        print(f"  ${abs(ratio):.2f} of revenue impact per kg CO2e change "
              f"(guardrail result WAS significant, so this ratio is meaningful).")
    else:
        print("  Guardrail CO2e result was NOT statistically significant -- computing a "
              "'$ per kg CO2e' ratio from a non-significant guardrail estimate would be "
              "misleading (dividing by a noisy near-zero number), so it's omitted here. "
              "The honest read is: this experiment did not detect a reliable change in "
              "total footprint, even though it detected a reliable increase in green-choice "
              "behavior among converters and a reliable drop in conversion.")

    print("\nWhat a PM would need to weigh:")
    print("  - The nudge reliably costs conversions (and therefore revenue).")
    print("  - It reliably increases green-alternative selection among people who DO convert.")
    print("  - It does NOT reliably reduce total footprint once the conversion loss is priced in --")
    print("    the conversion-loss and greener-choices effects appear to roughly offset each other.")
    print("  - Recommendation would hinge on whether the business values the *behavioral* signal")
    print("    (more people choosing green when given the option) even without a proven aggregate")
    print("    footprint reduction, and whether a redesigned nudge (e.g. less friction, easier")
    print("    swap) could keep the green-choice lift while recovering some of the conversion cost.")


if __name__ == "__main__":
    main()
