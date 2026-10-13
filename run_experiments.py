
"""Run the Airbnb price prediction experiment on the supplied dataset."""

import argparse
import time

import pandas as pd

import config
from src.data import load_data, split_data
from src.features import FeatureBuilder
from src.models import evaluate, get_model


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare Airbnb price prediction models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "rf", "xgb"],
        choices=["linear", "rf", "xgb"],
        help="Models to evaluate (default: linear rf xgb).",
    )
    args = parser.parse_args()

    df = load_data()
    train, test = split_data(df)

    # Fit preprocessing only on training data.
    builder = FeatureBuilder().fit(train)
    X_train = builder.transform(train)
    X_test = builder.transform(test)

    y_train = train["log_price"].to_numpy()
    y_test = test["log_price"].to_numpy()

    print(f"Dataset: {len(df):,} rows")
    print(f"Train/test: {len(train):,}/{len(test):,}")
    print(f"Feature matrix: {X_train.shape[1]:,} columns")
    print(f"Models: {', '.join(args.models)}")

    rows = []

    for name in args.models:
        print(f"\nTraining {name}...")
        start = time.perf_counter()

        model = get_model(name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        elapsed = time.perf_counter() - start
        metrics = evaluate(y_test, predictions)

        row = {
            "model": name,
            **metrics,
            "seconds": round(elapsed, 2),
        }
        rows.append(row)
        print(row)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows).sort_values("r2", ascending=False)

    results.to_csv(
        config.RESULTS_DIR / "model_comparison.csv",
        index=False,
    )
    results.to_csv(
        config.RESULTS_DIR / "results.csv",
        index=False,
    )

    print("\nResults written to:")
    print(config.RESULTS_DIR / "model_comparison.csv")
    print(config.RESULTS_DIR / "results.csv")


if __name__ == "__main__":
    main()
