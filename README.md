# Airbnb Listing Price Prediction

This project trains regression models to estimate Airbnb listing prices from listing details, location, amenities, and text. It compares Ridge regression, Random Forest, and XGBoost.

## Dataset

The project includes `data/Airbnb_Data.csv.gz`, containing 74,111 listings across New York City, Los Angeles, San Francisco, Washington, D.C., Chicago, and Boston. The target is the dataset’s existing `log_price` column. Metrics therefore describe prediction error in log-price units; they are not dollar errors.

## Workflow

1. Load the supplied compressed CSV and remove rows with an invalid target.
2. Make a reproducible 80/20 train/test split, stratified by city (seed 42).
3. Fit preprocessing on the training partition only: median imputation and scaling for numeric fields, one-hot encoding for categorical fields, and TF-IDF features for amenities and listing text.
4. Train the requested regression model or models.
5. Evaluate on the held-out test set with MAE, RMSE, and R².

## Results

The checked-in result files currently contain a Ridge regression run:

| Model | MAE (log price) | RMSE (log price) | R² | Runtime (s) |
|---|---:|---:|---:|---:|
| Ridge | 0.2857 | 0.3967 | 0.6992 | 4.21 |

The run used 59,288 training rows and 14,823 test rows. These figures are from the saved `results/model_comparison.csv`; they are not a comparison against Random Forest or XGBoost. Results can vary with library versions or changes to the data and configuration.

## Setup and usage

Use Python 3.9 or newer, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
