"""Leakage-safe feature construction for the supplied Airbnb dataset."""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import config


def _amenity_tokens(value):
    if not isinstance(value, str):
        return ""
    value = re.sub(r'[{}\[\]"]', " ", value)
    return " ".join(x.strip().lower().replace(" ", "_") for x in value.split(",") if x.strip())


class FeatureBuilder:
    """Fit preprocessing only on the training data, then transform any split."""
    def fit(self, df):
        self.num_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
        self.cat_cols = [c for c in config.CATEGORICAL_COLUMNS if c in df.columns]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, self.num_cols),
            ("cat", cat_pipe, self.cat_cols),
        ], sparse_threshold=0.3)
        self.preprocessor.fit(df)

        # Amenities are represented as a compact TF-IDF feature block.
        self.amenity_vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, min_df=10, max_features=300,
        )
        amenity_text = df.get("amenities", pd.Series("", index=df.index)).map(_amenity_tokens)
        self.amenity_vectorizer.fit(amenity_text)

        # Free-form descriptions are deliberately capped to keep the experiment reproducible.
        self.text_vectorizer = TfidfVectorizer(
            min_df=10, max_features=5000, ngram_range=(1, 2), stop_words="english",
        )
        text = (
            df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
            + " " +
            df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
        )
        self.text_vectorizer.fit(text)
        return self

    def transform(self, df):
        base = self.preprocessor.transform(df)
        amenity = self.amenity_vectorizer.transform(
            df.get("amenities", pd.Series("", index=df.index)).map(_amenity_tokens)
        )
        text = (
            df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
            + " " +
            df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
        )
        text_mat = self.text_vectorizer.transform(text)

        from scipy.sparse import hstack
        return hstack([base, amenity, text_mat], format="csr", dtype=np.float32)
