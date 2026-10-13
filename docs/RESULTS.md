# Airbnb Price Prediction — Results

## 1. Overview

This document summarises the performance of three regression models trained to predict Airbnb listing prices: Linear Regression, Random Forest, and XGBoost.

The models were evaluated using a held-out test set with Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).

## 2. Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.3056 | 0.4187 | 0.6588 |
| Random Forest | 0.2830 | 0.3952 | 0.6960 |
| XGBoost | 0.2843 | 0.3911 | 0.7022 |

**Note:** MAE and RMSE are measured on the log-price target scale. They should not be interpreted directly as errors in dollars.

## 3. Interpretation

- **Linear Regression:** Achieved an R² of 0.6588, providing a baseline for comparison.
- **Random Forest:** Achieved the lowest MAE of 0.2830, giving it the smallest average absolute error on the evaluated scale.
- **XGBoost:** Achieved the highest R² of 0.7022 and lowest RMSE of 0.3911.

XGBoost was selected as the overall model based on its R² and RMSE results. Random Forest performed slightly better on MAE, so XGBoost was not the best model on every metric.

## 4. Conclusion

The evaluation indicates that all three models capture useful patterns in Airbnb listing prices. XGBoost provides the strongest overall performance according to R² and RMSE, while Random Forest has the lowest MAE.

The models remain imperfect predictors. Their performance is limited by the features available in the dataset and by factors affecting listing prices that may not be represented in the data.

## 5. Limitations

- The dataset covers six US cities and may not generalise to other markets.
- Predictions depend on the quality and coverage of the available features.
- The evaluation metrics reported here are on the log-price scale.
- Predictive feature importance does not establish causal relationships.

These results should be interpreted alongside the project's evaluation graphs and implementation.
