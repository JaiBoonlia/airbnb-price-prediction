# Airbnb Price Prediction

## Overview

This project applies machine learning to estimate Airbnb listing prices using listing characteristics, location information, amenities, and text features. The dataset contains 74,111 listings across six US cities.

## Problem Statement

Airbnb prices vary according to property characteristics, room type, location, and other listing attributes. This project explores these patterns and compares regression models to estimate listing prices.

## Dataset

- **File:** `data/Airbnb_Data.csv.gz`
- **Records:** 74,111
- **Target:** `log_price`
- **Locations:** New York City, Los Angeles, San Francisco, Washington, D.C., Chicago, and Boston

See `data/README.md` for dataset information.

## Methodology

1. Load and inspect the dataset.
2. Explore listing distributions, prices, cities, room types, and numerical correlations.
3. Split the data into training and test sets.
4. Fit preprocessing on training data and transform the datasets.
5. Train and compare Linear Regression, Random Forest, and XGBoost.
6. Evaluate using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
7. Visualise predictions, residuals, model comparisons, and feature importance.

The target variable is `log_price`. MAE and RMSE reported on this target scale are not dollar-denominated errors.

## Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.3056 | 0.4187 | 0.6588 |
| Random Forest | 0.2830 | 0.3952 | 0.6960 |
| XGBoost | 0.2843 | 0.3911 | 0.7022 |

XGBoost achieved the highest R² and lowest RMSE, while Random Forest achieved the lowest MAE. These figures must match the final experiment outputs before submission.

## Installation

Python 3.10 or a compatible version is recommended.

Create and activate a virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Experiments

Run the full experiment:

```bash
python run_experiments.py
```

To select specific models:

```bash
python run_experiments.py --models linear rf xgb
```

The results are written to `results/model_comparison.csv` and `results/results.csv`.

## Repository Structure

```text
airbnb-price-prediction/
├── data/
│   ├── Airbnb_Data.csv.gz
│   ├── README.md
│   └── .gitkeep
├── docs/
│   ├── RESULTS.md
│   ├── make_writeup.py
│   └── writeup.pdf
├── results/
│   ├── EDA/
│   │   ├── average_price_by_city.png
│   │   ├── listings_by_city.png
│   │   ├── numeric_correlation_matrix.png
│   │   ├── price_distribution_across_cities.png
│   │   └── price_distribution_by_room_type.png
│   ├── Model/
│   │   ├── actual_vs_predicted_xgboost.png
│   │   ├── model_performance_comparison.png
│   │   ├── residuals_vs_actual_price.png
│   │   ├── xgboost_feature_importance.png
│   │   └── xgboost_prediction_error_distribution.png
│   ├── model_comparison.csv
│   └── results.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   └── models.py
├── config.py
├── requirements.txt
└── run_experiments.py
```

## Limitations

- The dataset covers six US cities and may not generalise to other markets.
- Prediction quality depends on data quality and the features available.
- Important pricing factors may be absent from the dataset.
- Feature importance describes predictive contribution, not causation.
- Predictions are estimates, not guaranteed market prices.

## Conclusion

This project demonstrates the use of exploratory data analysis and regression models for Airbnb price prediction. XGBoost performs best on R² and RMSE in the reported comparison, while Random Forest has the lowest MAE. Model results should be interpreted alongside the evaluation metrics and limitations.
