"""Statistical tests for section 4 of the report.

Run from the project folder to print the results:  python src/stats.py

Payouts are tested on the log scale (log10), as in the EDA. With tens of
thousands of claims almost any difference is "significant", so every test also
reports an effect size, which says how big the difference is.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import eda

STORMS = ["Hurricane Floyd", "Hurricane Matthew", "Hurricane Florence",
          "Hurricane Helene", "Unnamed event"]


def paid_claims(df):
    paid = df[df["paid"] == 1].copy()
    paid["log_payment"] = np.log10(paid["payment"])
    return paid


def eta_squared(groups):
    """Share of the variance in the outcome explained by the groups."""
    values = np.concatenate(groups)
    grand_mean = values.mean()
    between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    total = ((values - grand_mean) ** 2).sum()
    return between / total


def zone_anova(df):
    """Test 1: does mean log payout differ across the three zone groups?"""
    paid = paid_claims(df)
    groups = [paid.loc[paid["zone_group"] == z, "log_payment"].values for z in eda.ZONES]
    f_stat, p_value = stats.f_oneway(*groups)
    levene = stats.levene(*groups)
    tukey = stats.tukey_hsd(*groups)
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            diff = groups[i].mean() - groups[j].mean()
            pairs.append({
                "comparison": f"{eda.ZONES[i]} vs {eda.ZONES[j]}",
                "mean_log_diff": round(diff, 3),
                "payout_ratio": round(10 ** diff, 2),
                "p_value": tukey.pvalue[i, j],
            })
    return {
        "statistic": f_stat,
        "p_value": p_value,
        "effect": eta_squared(groups),
        "levene_p": levene.pvalue,
        "n": sum(len(g) for g in groups),
        "pairs": pd.DataFrame(pairs),
    }


def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def zone_chi_square(df):
    """Test 2: is whether a claim is paid related to its zone group?"""
    data = df[df["zone_group"].isin(eda.ZONES)]
    table = pd.crosstab(data["zone_group"], data["paid"])
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    rates = (data.groupby("zone_group")["paid"].mean() * 100).round(1)
    return {
        "statistic": chi2,
        "dof": dof,
        "p_value": p_value,
        "effect": cramers_v(table),
        "n": int(table.values.sum()),
        "min_expected": expected.min(),
        "paid_rate_pct": rates,
    }


def firm_t_test(df):
    """Test 3: do post-FIRM buildings get different payouts than pre-FIRM ones?"""
    paid = paid_claims(df)
    post = paid.loc[paid["post_firm"] == 1, "log_payment"]
    pre = paid.loc[paid["post_firm"] == 0, "log_payment"]
    t_stat, p_value = stats.ttest_ind(post, pre, equal_var=False)
    pooled_sd = np.sqrt(((len(post) - 1) * post.var() + (len(pre) - 1) * pre.var())
                        / (len(post) + len(pre) - 2))
    return {
        "statistic": t_stat,
        "p_value": p_value,
        "effect": (post.mean() - pre.mean()) / pooled_sd,
        "n_post": len(post),
        "n_pre": len(pre),
        "median_post": paid.loc[paid["post_firm"] == 1, "payment"].median(),
        "median_pre": paid.loc[paid["post_firm"] == 0, "payment"].median(),
    }


def depth_spearman(df):
    """Test 4: do deeper floods go with larger payouts?"""
    paid = paid_claims(df).dropna(subset=["water_depth"])
    rho, p_value = stats.spearmanr(paid["water_depth"], paid["payment"])
    return {"statistic": rho, "p_value": p_value, "effect": rho, "n": len(paid)}


def zone_by_storm(df):
    """Repeat the zone comparison inside single storms.

    Claims from one storm share its wind, rain and surge, so they are not
    independent. Comparing zones within a storm removes the storm mix.
    """
    paid = paid_claims(df)
    rows = []
    for storm in STORMS:
        part = paid[(paid["event"] == storm) & paid["zone_group"].isin(eda.ZONES)]
        groups = [part.loc[part["zone_group"] == z, "log_payment"].values for z in eda.ZONES]
        groups = [g for g in groups if len(g) >= 30]
        row = {"event": storm}
        for zone in eda.ZONES:
            values = part.loc[part["zone_group"] == zone, "payment"]
            row[zone] = round(values.median()) if len(values) >= 30 else np.nan
        row["eta_squared"] = round(eta_squared(groups), 3)
        row["p_value"] = stats.f_oneway(*groups).pvalue
        rows.append(row)
    return pd.DataFrame(rows).set_index("event")


def fig_zone_by_storm(table):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.27
    x = np.arange(len(table))
    for i, zone in enumerate(eda.ZONES):
        ax.bar(x + (i - 1) * width, table[zone], width, label=zone,
               color=eda.ZONE_COLORS[zone])
    ax.set_xticks(x, [name.replace("Hurricane ", "") for name in table.index])
    ax.yaxis.set_major_formatter(eda.DOLLARS)
    ax.set_ylabel("Median payout, 2025 dollars")
    ax.set_title("Median payout by zone within each storm")
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.grid(axis="y", color=eda.GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def main():
    df = eda.load()
    pd.set_option("display.width", 200, "display.max_columns", 20)
    anova = zone_anova(df)
    print("ANOVA", {k: v for k, v in anova.items() if k != "pairs"})
    print(anova["pairs"])
    print("Chi-square", zone_chi_square(df))
    print("t-test", firm_t_test(df))
    print("Spearman", depth_spearman(df))
    table = zone_by_storm(df)
    print(table)
    fig_zone_by_storm(table).savefig("figures/zone_by_storm.png", dpi=150)


if __name__ == "__main__":
    main()
