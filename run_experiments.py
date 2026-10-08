"""Run the Airbnb price prediction experiment on the supplied dataset."""
import argparse
import time
import pandas as pd
import config
from src.data import load_data, split_data
from src.features import FeatureBuilder
from src.models import evaluate, get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["ridge"], choices=["ridge", "rf", "xgb"])
    args = parser.parse_args()

    df = load_data()
    train, test = split_data(df)
    builder = FeatureBuilder().fit(train)
    X_train = builder.transform(train)
    X_test = builder.transform(test)
    y_train = train["log_price"].to_numpy()
    y_test = test["log_price"].to_numpy()

    rows = []
    print(f"Dataset: {len(df):,} rows")
    print(f"Train/test: {len(train):,}/{len(test):,}")
    print(f"Feature matrix: {X_train.shape[1]:,} columns")

    for name in args.models:
        start = time.time()
        model = get_model(name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred)
        row = {"model": name, **metrics, "seconds": round(time.time() - start, 2)}
        rows.append(row)
        print(row)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    result = pd.DataFrame(rows).sort_values("r2", ascending=False)
    result.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    result.to_csv(config.RESULTS_DIR / "results.csv", index=False)
    print("\nResults written to results/model_comparison.csv")


if __name__ == "__main__":
    main()
