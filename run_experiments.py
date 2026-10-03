"""Reproduce the experiments of Luo, Zhou & Zhou (2019), Tables 1-7.

    python run_experiments.py --table 1
    python run_experiments.py --table all

Results are written to results/tableN.csv and printed. Metrics for log / sqrt
labels are computed in the transformed label space, as in the paper.
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from src import data
from src.features import FeatureBuilder
from src.models import evaluate, get_model


def run(models, train_cities, test_cities, groups, label, threshold, text_mode="tfidf_bi", scatter=None):
    """Train on the (combined) train split of `train_cities`.

    Test data = the held-out test splits of the train cities, or - if
    `test_cities` is given (Berlin transfer) - the entire unseen city dataset.
    """
    parts, tests = [], []
    for city in train_cities:
        tr, _val, te = data.split(data.clean(data.load_city(city), threshold))
        parts.append(tr)
        tests.append(te)
    if test_cities:      # transfer setting: evaluate only on the unseen city
        tests = [data.clean(data.load_city(c), threshold) for c in test_cities]
    train_df = pd.concat(parts, ignore_index=True)
    test_df = pd.concat(tests, ignore_index=True)

    fb = FeatureBuilder(groups, text_mode=text_mode).fit(train_df)
    X_tr, X_te = fb.transform(train_df), fb.transform(test_df)
    y_tr = data.transform_label(train_df["price"].values, label)
    y_te = data.transform_label(test_df["price"].values, label)

    rows = []
    for name in models:
        t0 = time.time()
        model = get_model(name).fit(X_tr, y_tr)
        tr_m, te_m = evaluate(y_tr, model.predict(X_tr)), evaluate(y_te, model.predict(X_te))
        rows.append({
            "model": name, "train_data": "+".join(train_cities), "test_data": "+".join(test_cities) or "held-out",
            "feature": groups if text_mode == "tfidf_bi" or "T" not in groups else f"{groups}({text_mode})",
            "label": label, "threshold": threshold,
            "train_mse": tr_m["mse"], "train_r2": tr_m["r2"], "test_mse": te_m["mse"], "test_r2": te_m["r2"],
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  {name:7s} {groups:6s} {label:4s} test R2={te_m['r2']:.3f}")
        if scatter:
            _save_scatter(y_te, model.predict(X_te), scatter)
    return rows


def _save_scatter(y_true, y_pred, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.scatter(y_true, y_pred, s=2)
    plt.xlabel("Ground-truth label"); plt.ylabel("Predicted label"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


# ------------------------------------------------------------------ tables
def table1():   # continuous features, price thresholded at $500, NYC
    return run(["linear", "knn", "rf", "nn", "xgb"], ["nyc"], [], "C", "none", True)

def table2():   # XGBoost with label transformations (no thresholding)
    rows = []
    for groups, label in [("C", "none"), ("C", "sqrt"), ("C", "log"), ("C+I", "log")]:
        rows += run(["xgb"], ["nyc"], [], groups, label, False)
    return rows

def table3():   # text-feature variants (thresholded)
    rows = []
    for mode in ["count_uni", "tfidf_uni", "tfidf_bi"]:
        rows += run(["xgb"], ["nyc"], [], "T", "none", True, text_mode=mode)
    return rows

def table4():   # date / categorical / combined features
    rows = run(["xgb"], ["nyc"], [], "D", "none", True)
    rows += run(["xgb"], ["nyc"], [], "O", "none", True)
    rows += run(["xgb"], ["nyc"], [], "C+O+T", "none", True)
    rows += run(["xgb"], ["nyc"], [], "C+O+T", "log", False)
    return rows

def table5():   # Paris
    rows = run(["xgb"], ["paris"], [], "C+O+T", "none", True)
    rows += run(["xgb"], ["paris"], [], "C+O+T", "log", False)
    return rows

def table6():   # NN: individual vs combined training
    rows = []
    for cities in (["nyc"], ["paris"], ["nyc", "paris"]):
        rows += run(["nn"], cities, [], "C+O", "log", False)
    return rows

def table7():   # NN transfer to unseen Berlin
    rows = []
    for cities in (["nyc"], ["paris"], ["nyc", "paris"]):
        tag = "_".join(cities)
        rows += run(["nn"], cities, ["berlin"], "C+O", "log", False,
                    scatter=config.RESULTS_DIR / f"fig6_train_{tag}_test_berlin.png")
    return rows


TABLES = {1: table1, 2: table2, 3: table3, 4: table4, 5: table5, 6: table6, 7: table7}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="all", help="1-7 or 'all'")
    args = ap.parse_args()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    todo = list(TABLES) if args.table == "all" else [int(args.table)]
    for t in todo:
        print(f"== Table {t} ==")
        df = pd.DataFrame(TABLES[t]())
        df.to_csv(config.RESULTS_DIR / f"table{t}.csv", index=False)
        print(df.round(4).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
