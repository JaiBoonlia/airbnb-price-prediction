"""Dataset loading, cleaning and reproducible train/test splitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
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
    df = df[df["log_price"].notna()].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, seed=config.SEED):
    # Keep the city mix in both partitions so the evaluation is not dominated by one city.
    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=seed,
        stratify=df["city"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
