# Airbnb Price Prediction

A reproducible machine-learning project that predicts Airbnb listing price from the supplied Airbnb dataset.

## Dataset

The project uses `data/Airbnb_Data.csv.gz` directly. It contains 74,111 listings from six cities: NYC, LA, SF, DC, Chicago and Boston.

The dataset already provides `log_price`, so the experiment predicts `log_price` directly.

## Pipeline

1. Load and clean the supplied CSV.
2. Remove rows with missing target values.
3. Split the data into 80% training and 20% testing with city-stratified sampling.
4. Fit all preprocessing on the training split only.
5. Use numeric listing/review/location variables, categorical listing variables, amenities and TF-IDF text features.
6. Compare Ridge regression, Random Forest and XGBoost.
7. Report MAE, RMSE and R² on the held-out test set.

## Reproducibility

```bash
pip install -r requirements.txt
python run_experiments.py
```

Results are generated in `results/model_comparison.csv`.

## Important

The numerical results in this repository are generated from the supplied dataset by `run_experiments.py`. All reported metrics are generated directly from the supplied dataset.
