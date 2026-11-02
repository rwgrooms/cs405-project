"""Exploratory analysis: tables and figures for the report.

Run from the project folder to save the figures:  python src/eda.py
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#52514e"
GRID = "#e4e3df"

ZONES = ["Low risk", "Zone A", "Zone V"]
ZONE_COLORS = {"Low risk": AQUA, "Zone A": BLUE, "Zone V": ORANGE}

COUNTIES = {
    37013: "Beaufort", 37019: "Brunswick", 37021: "Buncombe", 37031: "Carteret",
    37049: "Craven", 37055: "Dare", 37065: "Edgecombe", 37095: "Hyde",
    37129: "New Hanover", 37133: "Onslow", 37137: "Pamlico", 37141: "Pender",
    37147: "Pitt", 37195: "Wilson",
}

PERIODS = [
    (1978, 1994, "1978-1994: before the 1994 reform act"),
    (1995, 2012, "1995-2012: higher coverage limits"),
    (2013, 2021, "2013-2021: after Biggert-Waters"),
    (2022, 2026, "2022-2026: Risk Rating 2.0"),
]

MISSING_COLUMNS = {
    "buildingPropertyValue": "Building value",
    "waterDepth": "Water depth",
    "elevationDifference": "Elevation difference",
    "floodWaterDuration": "Flood duration",
    "buildingDescriptionCode": "Building description",
    "floodEvent": "Event name",
    "ratedFloodZone": "Flood zone",
    "buildingDeductibleCode": "Deductible",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": GRAY,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


DOLLARS = FuncFormatter(lambda value, _: f"${value:,.0f}")
THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")


def load():
    return pd.read_csv("data/claims_clean.csv")


def load_raw():
    return pd.read_csv("data/nfip_claims_nc.csv", low_memory=False)


def add_period(df):
    df = df.copy()
    df["period"] = ""
    for start, end, label in PERIODS:
        df.loc[df["year"].between(start, end), "period"] = label
    return df


def summary(df, column):
    """Claim counts, paid rate and median payout for each group."""
    paid = df[df["paid"] == 1]
    table = pd.DataFrame({
        "claims": df.groupby(column).size(),
        "paid_rate_pct": (df.groupby(column)["paid"].mean() * 100).round(1),
        "median_payout": paid.groupby(column)["payment"].median().round(0),
        "total_paid_millions": (paid.groupby(column)["payment"].sum() / 1e6).round(1),
    })
    return table


def outliers(df):
    """Flag paid claims outside 1.5 IQR on the log scale."""
    log_pay = np.log10(df.loc[df["paid"] == 1, "payment"])
    q1, q3 = log_pay.quantile([0.25, 0.75])
    low = 10 ** (q1 - 1.5 * (q3 - q1))
    high = 10 ** (q3 + 1.5 * (q3 - q1))
    return {
        "low": low,
        "high": high,
        "below": int((log_pay < np.log10(low)).sum()),
        "above": int((log_pay > np.log10(high)).sum()),
    }


def fig_missing(raw):
    """Percent missing by decade for the columns with gaps."""
    decade = raw["yearOfLoss"] // 10 * 10
    table = raw[list(MISSING_COLUMNS)].isna().groupby(decade).mean() * 100
    table = table.T.rename(index=MISSING_COLUMNS)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(table.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [f"{d}s" for d in table.columns])
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            color = "white" if value > 55 else "#0b0b0b"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=9)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Percent of claims missing each value, by decade")
    fig.tight_layout()
    return fig


def fig_payment_hist(df):
    paid = df.loc[df["paid"] == 1, "payment"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.logspace(2, np.log10(paid.max()), 45)
    ax.hist(paid, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(DOLLARS)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.axvline(paid.median(), color=GRAY, linestyle="--", linewidth=1)
    ax.annotate(f"Median ${paid.median():,.0f}", (paid.median(), ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", color=GRAY)
    ax.set_xlabel("Building payout, 2025 dollars (log scale)")
    ax.set_ylabel("Paid claims")
    ax.set_title("Payouts span four orders of magnitude")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig_payment_by_zone(df):
    paid = df[(df["paid"] == 1) & df["zone_group"].isin(ZONES)]
    data = [paid.loc[paid["zone_group"] == z, "payment"] for z in ZONES]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    box = ax.boxplot(data, vert=False, tick_labels=ZONES, showfliers=False, whis=(5, 95),
                     patch_artist=True, widths=0.55,
                     medianprops={"color": "#0b0b0b", "linewidth": 1.5})
    for patch, zone in zip(box["boxes"], ZONES):
        patch.set_facecolor(ZONE_COLORS[zone])
        patch.set_edgecolor("white")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(DOLLARS)
    ax.set_xlabel("Building payout, 2025 dollars (log scale). Whiskers: 5th to 95th percentile")
    ax.set_title("Payout by flood zone group, paid claims")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig_payment_by_occupancy(df):
    paid = df[(df["paid"] == 1) & (df["occupancy"] != "Unknown")]
    order = paid.groupby("occupancy")["payment"].median().sort_values().index
    data = [paid.loc[paid["occupancy"] == o, "payment"] for o in order]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    box = ax.boxplot(data, vert=False, tick_labels=list(order), showfliers=False, whis=(5, 95),
                     patch_artist=True, widths=0.55,
                     medianprops={"color": "#0b0b0b", "linewidth": 1.5})
    for patch in box["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_edgecolor("white")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(DOLLARS)
    ax.set_xlabel("Building payout, 2025 dollars (log scale). Whiskers: 5th to 95th percentile")
    ax.set_title("Payout by building type, paid claims")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig_claims_by_year(df):
    counts = df.groupby("year").size()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(counts.index, counts.values, color=BLUE, width=0.8)
    labels = {1996: "Fran", 1999: "Floyd", 2003: "Isabel", 2011: "Irene",
              2016: "Matthew", 2018: "Florence", 2024: "Helene"}
    for year, name in labels.items():
        align = "right" if name == "Matthew" else "center"
        ax.annotate(name, (year, counts[year]), xytext=(0, 3), textcoords="offset points",
                    ha=align, color=GRAY, fontsize=9)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_ylabel("Claims")
    ax.set_title("Claims arrive in storm-driven spikes")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig_correlation(df):
    paid = df[df["paid"] == 1]
    columns = {
        "payment": "Payout", "coverage": "Coverage", "building_value": "Building value",
        "building_age": "Building age", "water_depth": "Water depth",
        "elevation_diff": "Elevation diff.", "deductible": "Deductible",
        "post_firm": "Post-FIRM", "elevated": "Elevated", "year": "Year",
    }
    corr = paid[list(columns)].rename(columns=columns).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) > 0.6 else "#0b0b0b"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Spearman correlation, paid claims")
    fig.tight_layout()
    return fig


def fig_map(df):
    """Bubble map of claims; locations are rounded in the source data."""
    points = df.dropna(subset=["latitude", "longitude"])
    points = points.groupby(["latitude", "longitude"]).size().reset_index(name="claims")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(points["longitude"], points["latitude"], s=points["claims"] / 30,
               color=BLUE, alpha=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where claims happen: mostly the coast")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def county_table(df):
    table = summary(df.dropna(subset=["county"]), "county")
    table = table.sort_values("total_paid_millions", ascending=False).head(10)
    table.index = [COUNTIES.get(int(code), str(int(code))) for code in table.index]
    table.index.name = "county"
    return table


def event_table(df):
    table = summary(df, "event")
    return table.sort_values("total_paid_millions", ascending=False).head(10)


def period_table(df):
    """Compare periods for all claims and for claims outside named storms."""
    df = add_period(df)
    table = summary(df, "period")
    unnamed = df[df["event"] == "Unnamed event"]
    table["median_payout_unnamed"] = summary(unnamed, "period")["median_payout"]
    return table


def main():
    df = load()
    raw = load_raw()
    figures = {
        "missing_by_decade": fig_missing(raw),
        "payment_histogram": fig_payment_hist(df),
        "payment_by_zone": fig_payment_by_zone(df),
        "payment_by_occupancy": fig_payment_by_occupancy(df),
        "claims_by_year": fig_claims_by_year(df),
        "correlation": fig_correlation(df),
        "claims_map": fig_map(df),
    }
    for name, fig in figures.items():
        fig.savefig(f"figures/{name}.png", dpi=150)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(summary(df, "zone_group"))
    print(summary(df, "occupancy"))
    print(period_table(df))
    print(event_table(df))
    print(county_table(df))
    print(outliers(df))


if __name__ == "__main__":
    main()
