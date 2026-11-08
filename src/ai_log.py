"""Ask the assistant about each major project decision and save the answers.

Run from the project folder:  python src/ai_log.py
Each prompt goes through the same LangChain chain the app uses. Answers are
saved to data/ai_log.json. Entries already in the file are skipped, so the
evaluation written for each answer is kept when the script runs again.
"""
import json
from datetime import date
from pathlib import Path

import assistant

LOG = Path("data/ai_log.json")

DECISIONS = [
    {
        "id": "missing",
        "topic": "Missing values",
        "context": (
            "109,593 NC claims. buildingPropertyValue is missing on many claims; "
            "when it is missing only 2% of claims were paid, against 90% when it "
            "is present. floodWaterDuration is 94% missing for 2020s claims. "
            "foundationType holds 0 on 76% of rows, and the data dictionary does "
            "not define 0. Older claims lack deductible and flood zone."
        ),
        "question": (
            "How should I handle these missing values? Should I drop incomplete "
            "rows, impute, or drop columns? The payout model will use paid claims only."
        ),
    },
    {
        "id": "water_depth",
        "topic": "Water depth units",
        "context": (
            "waterDepth has a median of 1 and the data dictionary says some values "
            "are in feet and some in inches, with no column saying which. Values of "
            "900 and above are placeholders. It is 33% missing in the 2020s."
        ),
        "question": (
            "Can I still use water depth to study payouts, and if so how? Should I "
            "try to convert the values to one unit?"
        ),
    },
    {
        "id": "outliers",
        "topic": "Outliers",
        "context": (
            "80,843 paid claims in 2025 dollars: median $13,822, mean $35,924, "
            "max $2,735,296. On a log scale the 1.5 IQR rule flags 50 claims above "
            "$703,514 and 637 below $291. Separately, 336 payments are larger than "
            "the listed building coverage."
        ),
        "question": (
            "Should I remove these outliers, and what should I do with the payments "
            "that exceed coverage?"
        ),
    },
    {
        "id": "inflation",
        "topic": "Inflation adjustment",
        "context": (
            "Claims run from 1978 to 2025. I plan to convert payouts to 2025 dollars "
            "using the annual average CPI-U (FRED series CPIAUCNS)."
        ),
        "question": (
            "Is CPI-U a good choice for adjusting flood payouts, or is there a "
            "better index? What should I say about it in the report?"
        ),
    },
    {
        "id": "tests",
        "topic": "Choice of statistical tests",
        "context": (
            "Research question: do claims in high-risk flood zones get paid more? "
            "Zones grouped as Low risk (20,237 claims), Zone A (77,417) and "
            "Zone V (10,264). Log10 payout is close to normal with similar spreads "
            "(SD 0.62 to 0.64). A few storms (Floyd, Matthew, Florence, Helene) "
            "produce most of the claims."
        ),
        "question": (
            "Which test should I use to compare payouts across the three zones, and "
            "what should I watch out for with this many claims coming from a few storms?"
        ),
    },
    {
        "id": "split",
        "topic": "Train and test split",
        "context": (
            "Model target is log10 payout on paid claims. Claims cluster in storms. "
            "Option 1: a random 80/20 split. Option 2: train on claims before "
            "Hurricane Florence (2018-09-01, 60,174 claims) and test on Florence and "
            "later (20,669 claims, including Helene)."
        ),
        "question": "Which split should I use to evaluate the payout models, and why?",
    },
    {
        "id": "models",
        "topic": "Model choice and features",
        "context": (
            "The proposal named Linear Regression, Random Forest and XGBoost. The "
            "course needs one baseline and one advanced model, and I want to keep "
            "to standard scikit-learn. Candidate features: coverage, building value, "
            "water depth, building age, deductible, latitude, longitude, post-FIRM, "
            "elevated, primary residence, rental, zone group, occupancy, floors, "
            "flood cause, year of loss, and elevation difference (51% missing)."
        ),
        "question": (
            "Which models should I keep, and are there features I should leave out "
            "given the time-based split?"
        ),
    },
    {
        "id": "results",
        "topic": "Reading the model results",
        "context": (
            "Test set (Florence onward). Median benchmark R2 -0.12. Linear "
            "Regression MAE 0.528, RMSE 0.653, R2 0.02 (log10 scale). Random Forest "
            "MAE 0.499, RMSE 0.619, R2 0.12. Test median payout $20,913 vs $12,114 "
            "in training. Top Random Forest importances: building value 0.23, "
            "coverage 0.14, building age 0.12, then longitude, latitude, water "
            "depth. Zone group is near the bottom."
        ),
        "question": (
            "How should I interpret these results? Can I conclude that flood zone "
            "does not affect payouts?"
        ),
    },
]


def main():
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    done = {entry["id"] for entry in log}
    key = assistant.get_key()
    if not key:
        raise SystemExit("No GOOGLE_API_KEY found.")
    for decision in DECISIONS:
        if decision["id"] in done:
            continue
        print("Asking:", decision["topic"])
        answer = assistant.ask(decision["question"], decision["context"], key)
        log.append({**decision, "model": assistant.MODEL,
                    "date": date.today().isoformat(), "response": answer,
                    "evaluation": "", "decision": "", "reason": ""})
        LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved", len(log), "entries to", LOG)


if __name__ == "__main__":
    main()
