# What Drives Flood Insurance Payouts in North Carolina?

CSC 405 individual data science project, Fall 2026. Robert Grooms.

## Objective

Identify the factors that predict how much the National Flood Insurance Program
pays on a building claim in North Carolina, and test whether claims in high-risk
flood zones are actually paid more than claims in low-risk zones.

## Dataset

FEMA NFIP Redacted Claims v3, filtered to North Carolina: 109,593 claims and
84 columns.

Source: https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3

## Project structure

```
data/      dataset and data dictionary
src/       source code; src/app.py is the Streamlit report
figures/   generated figures
report/    exported PDF report
```

## Running the code

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
```
