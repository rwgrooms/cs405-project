"""Clean the NC flood claims data.

Reads data/nfip_claims_nc.csv and writes:
  data/claims_clean.csv      one row per claim, decoded and in 2025 dollars
  data/missing_summary.csv   how missing values relate to payouts

Run from the project folder:  python src/clean.py
"""
import numpy as np
import pandas as pd

BASE_YEAR = 2025

OCCUPANCY = {
    1: "Single family", 11: "Single family",
    2: "2-4 units", 12: "2-4 units",
    3: "5+ units", 13: "5+ units",
    14: "Mobile home",
    15: "Condo", 16: "Condo",
    4: "Non-residential", 6: "Non-residential", 17: "Non-residential",
    18: "Non-residential", 19: "Non-residential",
}

FLOORS = {
    1: "One floor", 2: "Two floors", 3: "Three or more",
    4: "Split level", 5: "Mobile home", 6: "Townhouse",
}

CAUSE = {"0": "Other", "1": "Tidal", "2": "River or stream", "4": "Rainfall"}

DEDUCTIBLE = {
    "0": 500, "1": 1000, "2": 2000, "3": 3000, "4": 4000, "5": 5000,
    "9": 750, "A": 10000, "B": 15000, "C": 20000, "D": 25000, "E": 50000,
    "F": 1250, "G": 1500, "H": 200,
}

# Columns checked for missing values before deciding how to handle them.
MISSING_CHECK = [
    "buildingPropertyValue", "waterDepth", "elevationDifference",
    "floodWaterDuration", "buildingDescriptionCode", "floodEvent",
    "ratedFloodZone", "buildingDeductibleCode",
]


def zone_group(zone):
    """Group detailed flood zones into A, V and low-risk."""
    if pd.isna(zone):
        return "Unknown"
    zone = str(zone).upper()
    if zone.startswith("A"):
        return "Zone A"
    if zone.startswith("V"):
        return "Zone V"
    if zone[0] in ("X", "B", "C"):
        return "Low risk"
    return "Unknown"


def missing_summary(df):
    """Compare claims with and without a value for each column."""
    paid = df["netBuildingPaymentAmount"] > 0
    rows = []
    for col in MISSING_CHECK:
        missing = df[col].isna()
        rows.append({
            "column": col,
            "pct_missing": round(missing.mean() * 100, 1),
            "paid_rate_missing": round(paid[missing].mean(), 2),
            "paid_rate_present": round(paid[~missing].mean(), 2),
            "median_paid_missing": df.loc[paid & missing, "netBuildingPaymentAmount"].median(),
            "median_paid_present": df.loc[paid & ~missing, "netBuildingPaymentAmount"].median(),
        })
    return pd.DataFrame(rows)


def clean(df, cpi):
    # Negative payments are recoveries, not payouts, and there are only a few.
    df = df[df["netBuildingPaymentAmount"] >= 0].copy()

    # Put every dollar amount in 2025 dollars.
    cpi = cpi.set_index("year")["cpi"]
    factor = cpi[BASE_YEAR] / df["yearOfLoss"].map(cpi)

    out = pd.DataFrame({"id": df["id"], "year": df["yearOfLoss"]})
    out["date"] = df["dateOfLoss"].str[:10]
    out["county"] = df["countyCode"]

    # A few coordinates fall far outside North Carolina.
    in_nc = df["latitude"].between(33.5, 37) & df["longitude"].between(-85, -75)
    out["latitude"] = df["latitude"].where(in_nc)
    out["longitude"] = df["longitude"].where(in_nc)

    # A blank event name means the claim was not part of a named storm.
    out["event"] = df["floodEvent"].fillna("Unnamed event")

    out["zone_group"] = df["ratedFloodZone"].apply(zone_group)
    out["occupancy"] = df["occupancyType"].map(OCCUPANCY).fillna("Unknown")
    out["floors"] = df["numberOfFloorsInTheInsuredBuilding"].map(FLOORS).fillna("Unknown")
    out["cause"] = df["causeOfDamage"].map(CAUSE).fillna("Other")
    out["deductible"] = df["buildingDeductibleCode"].map(DEDUCTIBLE)

    out["post_firm"] = df["postFIRMConstructionIndicator"]
    out["elevated"] = df["elevatedBuildingIndicator"]
    out["primary_residence"] = df["primaryResidenceIndicator"]
    out["rental"] = df["rentalPropertyIndicator"]

    # Building age; a build date after the loss means the record was updated.
    built = pd.to_datetime(df["originalConstructionDate"], errors="coerce", utc=True).dt.year
    age = df["yearOfLoss"] - built
    out["building_age"] = age.where(age >= 0)

    # Values of 900 or more are placeholders, not measurements.
    out["water_depth"] = df["waterDepth"].where(df["waterDepth"] < 900)
    elevation = df["elevationDifference"]
    out["elevation_diff"] = elevation.where(elevation.abs() < 900)

    # Zero coverage is not a real policy amount.
    coverage = df["totalBuildingInsuranceCoverage"].replace(0, np.nan)
    out["coverage"] = (coverage * factor).round(2)
    out["building_value"] = (df["buildingPropertyValue"] * factor).round(2)

    out["payment_nominal"] = df["netBuildingPaymentAmount"]
    out["payment"] = (df["netBuildingPaymentAmount"] * factor).round(2)
    out["paid"] = (df["netBuildingPaymentAmount"] > 0).astype(int)
    return out


def main():
    raw = pd.read_csv("data/nfip_claims_nc.csv", low_memory=False)
    cpi = pd.read_csv("data/cpi_annual.csv")

    missing_summary(raw).to_csv("data/missing_summary.csv", index=False)
    claims = clean(raw, cpi)
    claims.to_csv("data/claims_clean.csv", index=False)
    print(f"{len(raw):,} raw claims, {len(claims):,} after cleaning")


if __name__ == "__main__":
    main()
