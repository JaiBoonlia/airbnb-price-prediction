"""Loading, cleaning, splitting and label transformation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def _parse_money(series: pd.Series) -> pd.Series:
    """'$1,200.00' -> 1200.0 (already-numeric columns pass through)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_city(city: str) -> pd.DataFrame:
    path = config.DATASETS[city]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see data/README.md")
    df = pd.read_csv(path, low_memory=False)
    for col in config.MONEY_COLUMNS:
        if col in df.columns:
            df[col] = _parse_money(df[col])
    return df


def clean(df: pd.DataFrame, threshold: bool) -> pd.DataFrame:
    """Drop listings with missing / non-positive price; optionally cap at $500."""
    df = df[df["price"].notna() & (df["price"] > 0)]
    if threshold:
        df = df[df["price"] <= config.PRICE_CAP]
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = config.SEED):
    """7:2:1 train/validation/test split."""
    tr, rest = train_test_split(df, train_size=config.SPLIT[0], random_state=seed)
    val_share = config.SPLIT[1] / (config.SPLIT[1] + config.SPLIT[2])
    val, te = train_test_split(rest, train_size=val_share, random_state=seed)
    return tr.reset_index(drop=True), val.reset_index(drop=True), te.reset_index(drop=True)


# ---- label transformation (Section 4.2 / Table 2) -------------------------
def transform_label(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return y
    if kind == "sqrt":
        return np.sqrt(y)
    if kind == "log":
        return np.log(y)
    raise ValueError(kind)


def inverse_label(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return y
    if kind == "sqrt":
        return np.square(y)
    if kind == "log":
        return np.exp(y)
    raise ValueError(kind)
