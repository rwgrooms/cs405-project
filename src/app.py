"""Streamlit report: what drives flood insurance payouts in North Carolina?

Run from the project folder:  streamlit run src/app.py
"""
import pandas as pd
import streamlit as st

import eda
import stats

st.set_page_config(page_title="NC Flood Insurance Payouts", layout="centered")


@st.cache_data
def load_data():
    return eda.load(), eda.load_raw()


df, raw = load_data()
paid = df[df["paid"] == 1]

st.title("What Drives Flood Insurance Payouts in North Carolina?")
st.caption("CSC 405 individual project, Fall 2026. Robert Grooms.")

st.markdown(
    "**Research question:** What factors predict how much the National Flood "
    "Insurance Program pays on a building claim in North Carolina, and do claims "
    "in high-risk flood zones actually get paid more than claims in low-risk zones?"
)

# Dataset overview
st.header("1. Dataset overview")
st.markdown(
    "The data is FEMA's [NFIP Redacted Claims v3]"
    "(https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3), filtered "
    f"to North Carolina: **{len(raw):,} claims and {raw.shape[1]} columns** from "
    f"{raw['yearOfLoss'].min()} to {raw['yearOfLoss'].max()}. Each row is one claim. "
    "The target is the net building payment, adjusted to 2025 dollars with the "
    "annual Consumer Price Index."
)
col1, col2, col3 = st.columns(3)
col1.metric("Claims", f"{len(df):,}")
col2.metric("Paid claims", f"{len(paid):,}")
col3.metric("Median payout", f"${paid['payment'].median():,.0f}")

# Data cleaning
st.header("2. Data cleaning")
st.subheader("Missing values")
st.pyplot(eda.fig_missing(raw))
st.markdown(
    "Gaps are tied to time, not scattered at random. Older claims lack the "
    "deductible, flood zone and building description. Flood duration is almost "
    "empty in the 2020s, so a simple rule such as dropping every incomplete row "
    "would remove whole eras from the analysis."
)
st.markdown("Missing values also relate to the outcome:")
st.dataframe(pd.read_csv("data/missing_summary.csv"), hide_index=True)
st.markdown(
    "The clearest case is building value: when it is missing, only 2% of claims "
    "were paid, against 90% when it is present. It appears to be recorded mainly "
    "when a claim is adjusted, so it is not missing at random. Since the payout "
    "model uses paid claims only, this column stays."
)

st.subheader("Cleaning decisions")
st.table(pd.DataFrame([
    ["Negative payments (4)", "Removed", "They are recoveries, not payouts."],
    ["Unpaid claims (26%)", "Kept for EDA and tests", "Whether a claim is paid is part of the question."],
    ["Flood duration", "Dropped", "94% missing in the 2020s."],
    ["Foundation type", "Dropped", "76% hold a 0, which the data dictionary does not define."],
    ["Pre-FIRM flag", "Dropped", "Contradicts the post-FIRM flag and drifts over time."],
    ["Event name", "Blank set to Unnamed event", "A blank means no named storm."],
    ["Flood zone", "Grouped into A, V and low risk", "Too many detailed codes; V is different from A."],
    ["Water depth", "Values of 900+ removed", "They are placeholders. Units also mix feet and inches."],
    ["Coordinates outside NC", "Removed", "They are data entry errors."],
    ["Zero coverage", "Treated as missing", "Not a real policy amount."],
    ["Dollar amounts", "Converted to 2025 dollars", "Claims span almost 50 years."],
], columns=["Item", "Decision", "Reason"]))

st.subheader("Outliers")
limits = eda.outliers(df)
st.markdown(
    f"On a log scale, the 1.5 IQR rule flags {limits['above']:,} paid claims above "
    f"\\${limits['high']:,.0f} and {limits['below']:,} below \\${limits['low']:,.0f}. "
    "These are kept: large payouts are real losses and are what insurance is for. "
    "Using a log scale limits their effect. Separately, 336 payments exceed the "
    "listed coverage. The data does not explain why, so they stay in and are "
    "treated as a data quality limit."
)

# Exploratory analysis
st.header("3. Exploratory analysis")

st.subheader("Payout distribution")
st.pyplot(eda.fig_payment_hist(df))
st.markdown(
    f"Most paid claims are modest, with a median of \\${paid['payment'].median():,.0f}, "
    f"but the mean is \\${paid['payment'].mean():,.0f} and the largest is "
    f"\\${paid['payment'].max():,.0f}. The shape is close to a bell curve only on a "
    "log scale, so models and tests use the log of payout."
)

st.subheader("Payouts by flood zone")
st.pyplot(eda.fig_payment_by_zone(df))
st.dataframe(eda.summary(df, "zone_group"))
st.markdown(
    "Low-risk claims have the highest median payout, slightly above Zone A, and "
    "Zone V has the lowest. High-risk zones account for most claims and most total "
    "dollars, but that reflects the number of claims, not larger payouts per claim."
)

st.subheader("Payouts by building type")
st.pyplot(eda.fig_payment_by_occupancy(df))
st.dataframe(eda.summary(df, "occupancy"))
st.markdown(
    "Larger buildings get larger payouts: 5+ unit buildings have about three "
    "times the median of single-family homes, which still make up most of the "
    "dollars paid."
)

st.subheader("Claims over time")
st.pyplot(eda.fig_claims_by_year(df))
st.markdown(
    "A handful of storms produce most claims, so claims from one storm are not "
    "independent. This matters for the tests and for how the models are validated."
)
st.dataframe(eda.event_table(df))
st.markdown(
    "Florence is the costliest event. Helene has far fewer claims but by far the "
    "highest median payout, and it hit inland western NC rather than the coast."
)

st.subheader("Where claims happen")
st.pyplot(eda.fig_map(df))
st.dataframe(eda.county_table(df))
st.markdown(
    "Claims concentrate along the coast and sounds, led by Dare and New Hanover. "
    "Buncombe is the inland exception: \\$65.5M of its \\$91.4M came from Helene."
)

st.subheader("Correlations")
st.pyplot(eda.fig_correlation(df))
st.markdown(
    "No single numeric column explains payouts: the strongest rank correlations "
    "with payout are below 0.2. Coverage is almost unrelated to payout, because "
    "coverage sets a ceiling and most claims are far below it. Deductible and year "
    "move together (0.63), so deductibles rose over time. Building age and post-FIRM "
    "are strongly linked (-0.64), so older buildings' results should not be read as "
    "two separate effects. Water depth is limited by its mixed units."
)

st.subheader("Do payouts change across policy periods?")
st.dataframe(eda.period_table(df))
st.markdown(
    "Median payout rises in every period, which at first looks like a policy "
    "effect. The last column tells a different story. Outside named storms, the "
    "median roughly doubled after 1994, when coverage limits were raised, then "
    "stayed flat at about \\$11,000 to \\$13,000. The later increases mostly reflect "
    "which storms hit, such as Florence and Helene, not policy changes. This is "
    "a comparison of medians, not proof of cause."
)

# Statistical analysis
st.header("4. Statistical analysis")
st.markdown(
    "Four tests check the patterns seen in the EDA, each at the 0.05 level. "
    "Payouts are tested on the log scale because raw payouts are heavily skewed. "
    "Two cautions apply to all four. First, with tens of thousands of claims even "
    "tiny differences give very small p-values, so each test also reports an "
    "effect size, which measures how big the difference is. Second, claims from "
    "the same storm are not independent, which makes p-values look stronger than "
    "they are. The last part repeats the main test inside single storms to check this."
)


def p_text(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


anova = stats.zone_anova(df)
st.subheader("Test 1: Payout by flood zone (one-way ANOVA)")
st.markdown(
    "**Null hypothesis:** mean log payout is the same in low-risk zones, Zone A "
    "and Zone V.  \n"
    "**Why this test:** it compares the means of one numeric outcome across "
    "three groups. Log payouts are close to normal, and the groups have similar "
    "spreads (standard deviations 0.62 to 0.64)."
)
col1, col2, col3 = st.columns(3)
col1.metric("F statistic", f"{anova['statistic']:.1f}")
col2.metric("p-value", p_text(anova["p_value"]))
col3.metric("Eta squared", f"{anova['effect']:.3f}")
st.markdown("Tukey's test compares each pair of zones:")
pairs = anova["pairs"].copy()
pairs["p_value"] = pairs["p_value"].map(p_text)
st.dataframe(pairs, hide_index=True)
st.markdown(
    "The null hypothesis is rejected, but zone explains only 0.3% of the variation "
    "in payouts (eta squared 0.003). Tukey's test shows where the difference is: "
    "low-risk and Zone A payouts are about the same (p = 0.29), and Zone V payouts "
    "are about 25% lower than both (the other two zones pay about 1.3 times as much). "
    "**Answer to the research question:** claims in high-risk zones do not get paid "
    "more. Zone A pays about the same as low-risk zones, and coastal Zone V pays less."
)

chi = stats.zone_chi_square(df)
st.subheader("Test 2: Whether a claim is paid, by flood zone (chi-square)")
st.markdown(
    "**Null hypothesis:** the share of claims that get paid is the same in every "
    "zone group.  \n"
    "**Why this test:** both variables are categories (paid or not, and zone "
    "group), and every expected cell count is far above 5 "
    f"(smallest {chi['min_expected']:,.0f})."
)
col1, col2, col3 = st.columns(3)
col1.metric("Chi-square", f"{chi['statistic']:.0f}")
col2.metric("p-value", p_text(chi["p_value"]))
col3.metric("Cramér's V", f"{chi['effect']:.3f}")
st.dataframe(chi["paid_rate_pct"].rename("paid rate (%)"))
st.markdown(
    "The null hypothesis is rejected. Zone A claims are paid most often (75.9%), "
    "then low-risk zones (70.4%), then Zone V (66.3%). The link is weak "
    "(Cramér's V 0.07), so zone says little about whether a claim gets paid."
)

firm = stats.firm_t_test(df)
st.subheader("Test 3: Post-FIRM vs pre-FIRM buildings (Welch t-test)")
st.markdown(
    "A FIRM is the flood map for a community. Post-FIRM buildings were built after "
    "the map existed, so they had to follow flood-zone building rules.  \n"
    "**Null hypothesis:** mean log payout is the same for post-FIRM and pre-FIRM "
    "buildings.  \n"
    "**Why this test:** it compares the means of two groups. The Welch version "
    "does not assume equal variances."
)
col1, col2, col3 = st.columns(3)
col1.metric("t statistic", f"{firm['statistic']:.1f}")
col2.metric("p-value", p_text(firm["p_value"]))
col3.metric("Cohen's d", f"{firm['effect']:.2f}")
st.markdown(
    f"The null hypothesis is rejected. Post-FIRM buildings have a median payout of "
    f"\\${firm['median_post']:,.0f}, compared with \\${firm['median_pre']:,.0f} for "
    "pre-FIRM buildings. The effect is small (d = -0.29). This matches the idea "
    "that newer flood rules reduce damage. But section 3 showed that post-FIRM "
    "buildings are also much newer (correlation -0.64), so the test cannot separate "
    "the effect of the building rules from building age."
)

depth = stats.depth_spearman(df)
st.subheader("Test 4: Water depth and payout (Spearman correlation)")
st.markdown(
    "**Null hypothesis:** there is no monotonic relationship between water depth "
    "and payout.  \n"
    "**Why this test:** Spearman uses ranks, so it does not need normal data. "
    "That matters here because water depth mixes feet and inches, and ranks are "
    "less affected by the mix than raw values."
)
col1, col2, col3 = st.columns(3)
col1.metric("Spearman rho", f"{depth['statistic']:.2f}")
col2.metric("p-value", p_text(depth["p_value"]))
col3.metric("Paid claims with depth", f"{depth['n']:,}")
st.markdown(
    "The null hypothesis is rejected: deeper water goes with larger payouts. The "
    "relationship is weak (rho 0.17), which is partly due to the mixed units. The "
    "true link is probably stronger than the data can show."
)

st.subheader("Check: zone comparison inside single storms")
storm_table = stats.zone_by_storm(df)
st.pyplot(stats.fig_zone_by_storm(storm_table))
shown = storm_table.copy()
shown["p_value"] = shown["p_value"].map(p_text)
st.dataframe(shown)
st.markdown(
    "Repeating Test 1 inside each storm removes the storm mix and the dependence "
    "between storms. Zone V has the lowest median in every storm. Low-risk and "
    "Zone A are close in Matthew, Florence and unnamed events, but they differ in "
    "both directions elsewhere: low-risk claims paid about a third more in Floyd, "
    "and Zone A paid about twice as much in Helene (Helene had too few Zone V "
    "claims to show). Effect sizes stay small "
    "(eta squared 0.004 to 0.014). The main result holds: high-risk zones do not "
    "get larger payouts, and zone explains little of the variation. That points "
    "to building and flood details, which the models in section 5 use."
)
