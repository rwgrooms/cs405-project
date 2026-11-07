"""Payout models for section 5 of the report.

Run from the project folder to print the results:  python src/models.py

Both models predict log10 payout for paid claims. They are trained on claims
before Hurricane Florence (September 2018) and tested on Florence onward, so
the test asks how well the model handles storms it has never seen.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import eda

SPLIT_DATE = "2018-09-01"
NUMERIC = ["coverage", "building_value", "water_depth", "building_age", "deductible",
           "latitude", "longitude", "post_firm", "elevated", "primary_residence", "rental"]
CATEGORICAL = ["zone_group", "occupancy", "floors", "cause"]


def split(df):
    paid = df[df["paid"] == 1].copy()
    paid["log_payment"] = np.log10(paid["payment"])
    train = paid[paid["date"] < SPLIT_DATE]
    test = paid[paid["date"] >= SPLIT_DATE]
    return train, test


def make_model(regressor):
    """Fill missing numbers with the median and one-hot encode categories."""
    prepare = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(prepare, regressor)


def scores(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "MAE_dollars": mean_absolute_error(10 ** actual, 10 ** predicted),
    }


def run(df):
    train, test = split(df)
    models = {
        "Linear Regression": make_model(LinearRegression()),
        "Random Forest": make_model(RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=405)),
    }
    rows, predictions = [], {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train[NUMERIC + CATEGORICAL], train["log_payment"])
        seconds = time.perf_counter() - start
        predicted = model.predict(test[NUMERIC + CATEGORICAL])
        predictions[name] = predicted
        rows.append({"model": name, **scores(test["log_payment"], predicted),
                     "train_seconds": seconds})
    baseline = np.full(len(test), train["log_payment"].median())
    rows.insert(0, {"model": "Median only (no model)",
                    **scores(test["log_payment"], baseline), "train_seconds": 0.0})
    results = pd.DataFrame(rows).set_index("model").round(3)
    return {"results": results, "models": models, "test": test,
            "predictions": predictions, "n_train": len(train), "n_test": len(test)}


def importance(forest_pipeline):
    """Random Forest importance, with one-hot columns added back together."""
    prepare, forest = forest_pipeline[0], forest_pipeline[-1]
    names = prepare.get_feature_names_out()
    values = pd.Series(forest.feature_importances_, index=names)
    groups = [name.split("__")[1] for name in names]
    for column in CATEGORICAL:
        groups = [column if g.startswith(column + "_") else g for g in groups]
    return values.groupby(groups).sum().sort_values()


def fig_importance(values):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(values.index, values.values, color=eda.BLUE)
    ax.set_xlabel("Share of importance")
    ax.set_title("What the Random Forest relies on")
    ax.grid(axis="x", color=eda.GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig_predicted(test, predicted):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(10 ** test["log_payment"], 10 ** predicted, s=3, alpha=0.2, color=eda.BLUE)
    low, high = 10, 10 ** test["log_payment"].max()
    ax.plot([low, high], [low, high], color=eda.GRAY, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(eda.DOLLARS)
    ax.yaxis.set_major_formatter(eda.DOLLARS)
    ax.set_xlabel("Actual payout (log scale)")
    ax.set_ylabel("Predicted payout (log scale)")
    ax.set_title("Random Forest on the test storms: predicted vs actual")
    ax.grid(color=eda.GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def main():
    output = run(eda.load())
    pd.set_option("display.width", 200)
    print(f"train {output['n_train']:,}  test {output['n_test']:,}")
    print(output["results"])
    values = importance(output["models"]["Random Forest"])
    print(values.sort_values(ascending=False).round(3))
    fig_importance(values).savefig("figures/rf_importance.png", dpi=150)
    fig_predicted(output["test"], output["predictions"]["Random Forest"]).savefig(
        "figures/rf_predicted.png", dpi=150)


if __name__ == "__main__":
    main()
