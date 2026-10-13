# Airbnb Price Prediction

## Project Overview

This project uses machine learning to predict Airbnb listing prices using listing characteristics and location information from six major US cities: New York City, Los Angeles, San Francisco, Washington, D.C., Chicago, and Boston.

The project explores the dataset, prepares features, trains regression models, and compares their predictive performance to identify a suitable model for price prediction.

## Problem Statement

Airbnb prices vary according to factors such as room type, property characteristics, and city. The goal of this project is to identify patterns in listing prices and build machine learning models that can estimate prices from the available listing features.

## Dataset

- **File:** `data/Airbnb_Data.csv.gz`
- **Number of records:** 74,111
- **Target variable:** `log_price`
- **Locations:** Six US cities

The dataset contains listing information used for exploratory data analysis, feature engineering, and regression modelling.

## Methodology

The project follows these steps:

1. Load and inspect the dataset.
2. Perform exploratory data analysis (EDA).
3. Examine price distributions, city-level differences, and relationships between numerical features.
4. Prepare the features and target variable for modelling.
5. Train and evaluate Linear Regression, Random Forest, and XGBoost models.
6. Compare the models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).
7. Visualise model predictions, residuals, and feature importance.

## Model Performance

The following results are reported on the held-out test set. MAE and RMSE are measured on the `log_price` scale.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.3056 | 0.4187 | 0.6588 |
| Random Forest | 0.2830 | 0.3952 | 0.6960 |
| XGBoost | 0.2843 | 0.3911 | 0.7022 |

XGBoost achieved the highest R² and lowest RMSE, while Random Forest achieved the lowest MAE. XGBoost was selected as the overall model because it performed best on two of the three evaluation metrics.

## Exploratory Data Analysis

The project includes visualisations covering:

- Number of listings by city.
- Average listing price by city.
- Price distributions across cities.
- Price distributions by room type.
- Correlations between numerical variables.

## Model Visualisations

The model evaluation visualisations include:

- Model performance comparison.
- Actual versus predicted prices.
- Prediction error distribution.
- Residuals versus actual prices.
- XGBoost feature importance.

Feature importance indicates which features contribute to the model's predictions. It does not establish that those features directly cause prices to change.

## Project Structure

```text
airbnb-price-prediction/
├── data/
│   ├── Airbnb_Data.csv.gz
│   └── README.md
├── docs/
│   ├── RESULTS.md
│   ├── make_writeup.py
│   └── writeup.pdf
├── results/
│   ├── EDA/
│   ├── Model/
│   ├── model_comparison.csv
│   └── results.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   └── models.py
├── config.py
├── requirements.txt
├── run_experiments.py
└── README.md
```

## Installation

Python 3.10 or a compatible version is recommended. Use the following commands from the project root.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

From the project root, run:

```bash
python run_experiments.py
```

The script runs the project's experiment pipeline. The exact outputs depend on the implementation and configuration in the repository.

## Limitations

- The dataset represents a specific set of US cities and may not generalise to other locations.
- Model performance depends on the available features and data quality.
- Listing prices can be influenced by factors not captured in the dataset.
- The evaluation metrics are reported on the log-price scale and should not be interpreted directly as dollar errors.
- Feature importance describes predictive contribution, not causation.

## Conclusion

This project demonstrates how exploratory data analysis and supervised regression can be used to study Airbnb listing prices. Among the evaluated models, XGBoost achieved the highest R² of 0.7022 and the lowest RMSE of 0.3911, while Random Forest achieved the lowest MAE of 0.2830.

The results show that machine learning can capture useful patterns in listing prices, although predictions remain estimates rather than guaranteed market prices.