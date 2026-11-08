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

Payouts are adjusted to 2025 dollars with the annual average of the Consumer
Price Index (CPI-U, not seasonally adjusted):
https://fred.stlouisfed.org/series/CPIAUCNS

## Project structure

```
data/      dataset, data dictionary, CPI series and the AI decision log
src/       source code; src/app.py is the Streamlit report
figures/   generated figures
report/    exported PDF report
```

| File | Purpose |
|---|---|
| src/clean.py | Cleans the raw claims and writes data/claims_clean.csv |
| src/eda.py | Exploratory tables and figures |
| src/stats.py | Hypothesis tests |
| src/models.py | Linear Regression and Random Forest payout models |
| src/assistant.py | LLM assistant: LangChain and the Gemini API |
| src/ai_log.py | Sends each major project decision to the assistant and saves the answers to data/ai_log.json |
| src/app.py | Streamlit report |

## Running the code

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/clean.py
streamlit run src/app.py
```

`python src/eda.py`, `python src/stats.py` and `python src/models.py` can also be
run on their own to print their results and save the figures.

## AI assistant

The report's AI section uses Gemini through LangChain. The app runs without a
key, with the assistant turned off. To turn it on, get a key from
https://aistudio.google.com/apikey and either type it into the sidebar or put it
in a `.env` file in the project folder:

```
GOOGLE_API_KEY=your-key
```

`.env` and `.streamlit/secrets.toml` are ignored by git, so the key is never
committed.
