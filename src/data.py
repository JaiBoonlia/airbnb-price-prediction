
"""Dataset loading, cleaning and reproducible train/test splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Load the dataset and remove rows with invalid target values."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {"log_price", "city"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["log_price"] = pd.to_numeric(df["log_price"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["log_price", "city"])
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")

    return df


def split_data(df: pd.DataFrame, seed=config.SEED):
    """Create a reproducible train/test split stratified by city."""
    city_counts = df["city"].value_counts()

    if len(city_counts) < 2 or (city_counts < 2).any():
        raise ValueError(
            "Each city must contain at least two valid rows for stratified splitting."
        )

    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=seed,
        stratify=df["city"],
    )

    return train.reset_index(drop=True), test.reset_index(drop=True)
