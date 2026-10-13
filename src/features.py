
"""Leakage-safe feature construction for the supplied Airbnb dataset."""

import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def _amenity_tokens(value):
    """Normalize an amenities string into whitespace-separated tokens."""
    if not isinstance(value, str):
        return ""

    value = re.sub(r'[{}\[\]"]', " ", value)
    return " ".join(
        item.strip().lower().replace(" ", "_")
        for item in value.split(",")
        if item.strip()
    )


def _text_column(df, column):
    """Return a text column, using empty strings when the column is absent."""
    return df.get(column, pd.Series("", index=df.index)).fillna("").astype(str)


def _listing_text(df):
    """Combine listing names and descriptions."""
    return _text_column(df, "name") + " " + _text_column(df, "description")


class FeatureBuilder:
    """Fit preprocessing on training data and transform other data splits."""

    def fit(self, df):
        self.num_cols = [
            column for column in config.NUMERIC_COLUMNS if column in df.columns
        ]
        self.cat_cols = [
            column for column in config.CATEGORICAL_COLUMNS if column in df.columns
        ]

        transformers = []

        if self.num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.num_cols))

        if self.cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])
            transformers.append(("cat", cat_pipe, self.cat_cols))

        if not transformers:
            raise ValueError("No configured numeric or categorical features were found.")

        self.preprocessor = ColumnTransformer(
            transformers,
            sparse_threshold=0.3,
        )
        self.preprocessor.fit(df)

        amenity_text = _text_column(df, "amenities").map(_amenity_tokens)

        self.amenity_vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            min_df=10,
            max_features=300,
        )
        self.amenity_vectorizer.fit(amenity_text)

        text = _listing_text(df)
        self.text_vectorizer = TfidfVectorizer(
            min_df=10,
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.text_vectorizer.fit(text)

        return self

    def transform(self, df):
        """Transform a data split using the already-fitted training pipelines."""
        base = self.preprocessor.transform(df)

        amenity_text = _text_column(df, "amenities").map(_amenity_tokens)
        amenity = self.amenity_vectorizer.transform(amenity_text)

        text = _listing_text(df)
        text_matrix = self.text_vectorizer.transform(text)

        return hstack(
            [base, amenity, text_matrix],
            format="csr",
            dtype=np.float32,
        )
