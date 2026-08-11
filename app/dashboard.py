"""
Interactive results dashboard for the CarbonCart A/B testing project.

Run locally with:
    streamlit run app/dashboard.py

Deploy for free at https://share.streamlit.io (Streamlit Community Cloud)
by pointing it at this file in your GitHub repo.

DESIGN NOTE: all numbers below are hardcoded, verified summary statistics
copied directly from real runs of the pipeline scripts in src/ (see
docs/REPORT.md for the full narrative and methodology behind each number).
This is intentional, not a shortcut: a results dashboard should present
already-computed, already-verified findings, not re-run a 16,800-row
simulation on every page load. If you want to regenerate these numbers
yourself, run the scripts in src/ per the README.
"""

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="CarbonCart: A/B Test Results",
    page_icon="🛒",
    layout="wide",
)

GREEN = "#2E7D32"
RED = "#C62828"
GRAY = "#757575"
BLUE = "#1565C0"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛒 CarbonCart: A Carbon-Nudge A/B Testing Case Study")
st.markdown(
    "A simulated e-commerce checkout experiment testing whether a real-time "
    "carbon footprint estimate changes shopper behavior, built on **real "
    "emission factor data** from the [Climatiq API](https://www.climatiq.io/) "
    "and a full randomized experimentation pipeline: power analysis, CUPED, "
    "sequential testing correction, multi-variant FDR correction, and a "
    "causal-inference comparison against biased observational data."
)
st.markdown(
    "[📄 Full technical write-up](https://github.com/YOUR-USERNAME/YOUR-REPO/blob/main/docs/REPORT.md) &nbsp;|&nbsp; "
    "[💻 Source code](https://github.com/YOUR-USERNAME/YOUR-REPO)"
)
st.caption("Replace YOUR-USERNAME/YOUR-REPO above with your actual GitHub path once the repo is live.")

st.divider()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversion rate", "-2.45pp", "p = 0.0006 (significant)", delta_color="inverse")
col2.metric("Green-alt selection", "+8.82pp", "p < 0.0001 (significant)", delta_color="normal")
col3.metric("Guardrail: footprint/user", "-0.71 kg", "p = 0.39 (not significant)", delta_color="off")
col4.metric("Sample size", "16,800 users", "sized via power analysis")

st.divider()

tabs = st.tabs([
    "A/B Test Results",
    "CUPED",
    "Sequential Testing",
    "Multi-Variant & FDR",
    "Causal Inference",
    "Guardrail Trade-off",
])

# ---------------------------------------------------------------------------
# Tab 1: Primary A/B test results
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Primary metrics: control vs. treatment")
    st.markdown(
        "16,800 simulated users were randomly split 50/50 between control (no carbon "
        "estimate shown) and treatment (carbon estimate + option to switch to a "
        "lower-carbon alternative)."
    )

    metrics = ["Conversion rate", "Green-alt rate\n(among converters)"]
    control_vals = [0.7026, 0.1582]
    treatment_vals = [0.6781, 0.2464]
    control_err = [0.0384 - 0.0245, 0.1028 - 0.0882]  # half-width approximations for display
    treatment_err = control_err

    fig = go.Figure()
    fig.add_bar(name="Control", x=metrics, y=control_vals, marker_color=GRAY)
    fig.add_bar(name="Treatment", x=metrics, y=treatment_vals, marker_color=GREEN)
    fig.update_layout(
        barmode="group",
        yaxis_title="Rate",
        yaxis_tickformat=".0%",
        title="Conversion & green-alternative selection rates",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Conversion rate**: 70.26% (control) → 67.81% (treatment), "
            "a difference of **-2.45pp**, 95% CI [-3.84pp, -1.05pp], **p = 0.0006**."
        )
    with c2:
        st.markdown(
            "**Green-alt selection**: 15.82% → 24.64% among converters, "
            "a difference of **+8.82pp**, 95% CI [+7.36pp, +10.28pp], **p < 0.0001**."
        )

    st.info(
        "Reading this together: the nudge reliably works as intended, making buyers "
        "choose lower-carbon options far more often, but it also reliably costs "
        "conversions. See the Guardrail tab for whether that trade-off nets out."
    )

# ---------------------------------------------------------------------------
# Tab 2: CUPED
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("CUPED variance reduction")
    st.markdown(
        "Using `historical_avg_spend` as a pre-experiment covariate to reduce noise "
        "in the conversion estimate ([Deng, Xu, Kohavi & Walker, 2013](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf))."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(
            x=["Without CUPED", "With CUPED"],
            y=[0.213754, 0.212312],
            marker_color=[GRAY, GREEN],
        )
        fig.update_layout(title="Outcome variance", yaxis_title="Variance")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Variance reduction", "0.67%")
        st.metric("Covariate correlation", "0.0821")
        st.markdown(
            "The reduction is small because `historical_avg_spend` only weakly predicts "
            "conversion in this simulation, CUPED's benefit is bounded by how strongly "
            "the chosen covariate actually correlates with the outcome. The effect "
            "estimate barely moved (-0.02446 → -0.02462), exactly as expected: CUPED "
            "changes precision, not the estimate itself."
        )

# ---------------------------------------------------------------------------
# Tab 3: Sequential testing
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Sequential testing / peeking correction")
    st.markdown(
        "What happens if results are checked repeatedly as data accumulates (e.g., "
        "daily) instead of once at a fixed sample size?"
    )

    looks = list(range(1, 11))
    zstats = [-1.839, -2.668, -3.024, -3.360, -3.202, -3.211, -3.548, -3.223, -3.226, -3.429]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=looks, y=zstats, mode="lines+markers", name="z-statistic", line=dict(color=BLUE)))
    fig.add_hline(y=-1.96, line_dash="dash", line_color=GRAY, annotation_text="Naive threshold (|z|=1.96)")
    fig.add_hline(y=1.96, line_dash="dash", line_color=GRAY)
    fig.add_hline(y=-2.582, line_dash="dot", line_color=RED, annotation_text="Corrected threshold (|z|=2.582)")
    fig.add_hline(y=2.582, line_dash="dot", line_color=RED)
    fig.update_layout(title="Z-statistic at each of 10 cumulative daily looks", xaxis_title="Look #", yaxis_title="z-statistic")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Nominal alpha", "5%")
    c2.metric("Naive false-positive rate (10 looks)", "20.3%", "4x inflation", delta_color="inverse")
    c3.metric("Corrected threshold", "|z| > 2.582", "vs. 1.960 uncorrected")

    st.success(
        "The real result clears the corrected threshold too (z = -2.668 at look 2), "
        "so the conclusion holds up under valid sequential testing, this isn't just a "
        "naive-peeking artifact."
    )

# ---------------------------------------------------------------------------
# Tab 4: Multi-variant + FDR
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Testing 3 nudge variants simultaneously")
    st.markdown(
        "Splitting the same traffic across 4 arms (control + 3 designs) instead of 2 "
        "reduces power per arm, this section shows why that matters."
    )

    variants = ["Control", "Plain estimate", "Green alternative", "Social proof"]
    conv_rates = [0.6952, 0.6958, 0.6895, 0.6974]
    green_rates = [0.1493, 0.2083, 0.2662, 0.2278]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=variants, y=conv_rates, marker_color=[GRAY, BLUE, BLUE, BLUE]))
        fig.update_layout(title="Conversion rate by variant", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(x=variants, y=green_rates, marker_color=[GRAY, GREEN, GREEN, GREEN]))
        fig.update_layout(title="Green-alt rate by variant (among converters)", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "None of the 3 variants showed a statistically significant conversion "
        "difference from control, naively or after Benjamini-Hochberg FDR correction "
        "(all p-values between 0.57 and 0.96). With ~4,200 users per arm, well under "
        "the ~8,400 the power analysis said was needed per group, this is an "
        "underpowered comparison by construction, not a verdict on any variant."
    )

# ---------------------------------------------------------------------------
# Tab 5: Causal inference
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Causal inference: RCT vs. biased observational data")
    st.markdown(
        "If this same effect had to be estimated from observational (non-randomized) "
        "data instead of a proper A/B test, how far off would the naive conclusion be?"
    )

    methods = ["True effect\n(known)", "Naive\nobservational", "Propensity-\nmatched", "Randomized\n(RCT)"]
    estimates = [-0.0200, 0.0161, -0.0293, -0.0245]
    colors = [GRAY, RED, BLUE, GREEN]

    fig = go.Figure(go.Bar(x=methods, y=estimates, marker_color=colors))
    fig.add_hline(y=-0.02, line_dash="dash", line_color=GRAY, annotation_text="True effect")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="Estimated treatment effect on conversion, by method", yaxis_title="Effect size")
    st.plotly_chart(fig, use_container_width=True)

    st.error(
        "The naive observational estimate doesn't just add noise, it reverses the "
        "sign entirely (+1.61pp reported vs. a true -2.00pp effect), because "
        "eco-conscious users both convert more AND were more likely to see the nudge."
    )
    st.success(
        "Propensity-score matching (logistic regression + nearest-neighbor matching, "
        "scikit-learn) recovers -2.93pp, correctly signed and within its 95% CI "
        "[-4.29pp, -1.57pp] of the true effect, a 74% reduction in bias vs. the naive estimate."
    )

# ---------------------------------------------------------------------------
# Tab 6: Guardrail trade-off
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Guardrail trade-off: should this ship?")

    rows = [
        ("Conversion rate", -0.0245, True),
        ("Revenue / user ($)", -1.24, False),
        ("Guardrail CO2e / user (kg)", -0.71, False),
        ("Green-alt uptake", 0.0882, True),
    ]
    labels = [r[0] for r in rows]
    diffs = [r[1] for r in rows]
    sig_colors = [GREEN if r[2] else GRAY for r in rows]

    fig = go.Figure(go.Bar(x=labels, y=diffs, marker_color=sig_colors))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="Treatment effect across all metrics (green = statistically significant)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**What a decision-maker would weigh:**
- The nudge **reliably costs conversions** (p = 0.0006), and directionally revenue, though the
  revenue estimate itself isn't statistically significant at this sample size (p = 0.14).
- It **reliably increases green-alternative selection** among people who do convert (p < 0.0001).
- It does **not reliably reduce total footprint** once the conversion loss is priced in (p = 0.39),
  the conversion-loss and greener-choice effects appear to roughly offset each other.
- A recommendation would hinge on whether the business values the behavioral signal on its own,
  and whether a redesigned, lower-friction nudge could preserve the green-choice lift while
  recovering some of the conversion cost.
        """
    )

st.divider()
st.caption(
    "Built with Python, pandas, statsmodels, scikit-learn, and the Climatiq API. "
    "All figures verified against real script output, see docs/REPORT.md for full methodology and limitations."
)
