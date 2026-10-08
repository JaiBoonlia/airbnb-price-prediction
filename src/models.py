"""Regression models and evaluation metrics."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


def get_model(name):
    if name == "ridge":
        return Ridge(alpha=config.RIDGE_ALPHA)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            random_state=config.SEED,
            n_jobs=-1,
        )
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=config.SEED,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate(y_true, y_pred):
    return {
        "mae_log_price": float(mean_absolute_error(y_true, y_pred)),
        "rmse_log_price": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }
