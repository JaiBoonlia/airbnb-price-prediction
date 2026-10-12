# Experimental Results

## Evaluation setup

- Dataset: `data/Airbnb_Data.csv.gz`
- Rows used: 74,111
- Train/test split: 59,288 / 14,823 rows
- Split strategy: 80/20, stratified by city
- Random seed: 42
- Target: `log_price`

## Recorded baseline

| Model | MAE (log price) | RMSE (log price) | R² |
|---|---:|---:|---:|
| Ridge regression | 0.2857 | 0.3967 | 0.6992 |

The metrics above are the recorded Ridge run on the supplied dataset. They are not copied from an external paper. MAE and RMSE are measured on the log-transformed target scale, so they should not be described as dollar errors.

## Reproduce

```bash
python run_experiments.py --models ridge
```

This rewrites `results/model_comparison.csv` and `results/results.csv`. Runtime varies by machine. Random Forest and XGBoost are available as optional experiments but do not have results in the checked-in baseline table unless separately run.

## Exploratory figures

The following figures are generated from the supplied dataset:

- `figures/listings_by_city.png`
- `figures/log_price_distribution.png`
- `figures/median_price_by_city.png`
- `figures/numeric_correlation_matrix.png`

These figures are descriptive; correlation does not imply causation.
