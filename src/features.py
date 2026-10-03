"""Feature engineering: continuous, interaction, categorical(+list), text, date.

Everything is *fitted on the training split only* and then applied to
validation/test/other-city data, so there is no leakage.

Feature groups (matching the paper's notation):
    C = continuous, I = 2-degree interaction terms of C, O = categorical,
    T = text (tf-idf -> SVD), D = date
Pass them as e.g. "C+O+T".
"""
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

import config

TEXT_MODES = {
    "count_uni": lambda: CountVectorizer(ngram_range=(1, 1), min_df=5, max_features=20000, stop_words="english"),
    "tfidf_uni": lambda: TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_features=20000, stop_words="english"),
    "tfidf_bi": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000, stop_words="english"),
}


def _tokenize_list(value) -> list:
    """'{TV,Wifi,"Air conditioning"}' or "['email', 'phone']" -> tokens."""
    if not isinstance(value, str):
        return []
    parts = re.split(r",", re.sub(r"[\{\}\[\]\"']", "", value))
    return [p.strip().lower() for p in parts if p.strip()]


class FeatureBuilder:
    def __init__(self, groups: str = "C", text_mode: str = "tfidf_bi", corr_threshold: float = config.CORR_THRESHOLD):
        self.groups = set(groups.split("+"))
        unknown = self.groups - {"C", "I", "O", "T", "D"}
        if unknown:
            raise ValueError(f"unknown feature groups: {unknown}")
        if "I" in self.groups and "C" not in self.groups:
            raise ValueError("'I' (interaction) requires 'C'")
        self.text_mode = text_mode
        self.corr_threshold = corr_threshold

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        if "C" in self.groups:
            self._fit_continuous(df)
        if "O" in self.groups:
            self._fit_categorical(df)
        if "T" in self.groups:
            self._fit_text(df)
        if "D" in self.groups:
            self._fit_date(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        if "C" in self.groups:
            blocks.append(self._tf_continuous(df))
        if "O" in self.groups:
            blocks.append(self._tf_categorical(df))
        if "T" in self.groups:
            blocks.append(self._tf_text(df))
        if "D" in self.groups:
            blocks.append(self._tf_date(df))
        return np.hstack(blocks).astype(np.float32)

    # ----------------------------------------------------------- continuous
    def _numeric(self, df, cols):
        return df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")

    def _fit_continuous(self, df):
        cols = [c for c in config.CONTINUOUS if c in df.columns]
        num = self._numeric(df, cols)
        cols = [c for c in cols if num[c].notna().any()]          # drop all-null columns
        num = num[cols]
        # drop one feature of every highly-correlated pair (Fig. 3 in the paper)
        corr = num.corr().abs()
        drop = set()
        for i, a in enumerate(cols):
            if a in drop:
                continue
            for b in cols[i + 1:]:
                if b not in drop and corr.loc[a, b] > self.corr_threshold:
                    drop.add(b)
        self.cont_cols = [c for c in cols if c not in drop]
        num = num[self.cont_cols]
        self.zero_fill = [c for c in config.PRICE_FILL_ZERO if c in self.cont_cols]
        num[self.zero_fill] = num[self.zero_fill].fillna(0)
        self.cont_means = num.mean()
        filled = num.fillna(self.cont_means)
        self.scaler = StandardScaler().fit(filled)
        self.poly = None
        if "I" in self.groups:
            self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self.poly.fit(self.scaler.transform(filled))

    def _tf_continuous(self, df):
        num = self._numeric(df, self.cont_cols)
        num[self.zero_fill] = num[self.zero_fill].fillna(0)
        z = self.scaler.transform(num.fillna(self.cont_means))
        return self.poly.transform(z) if self.poly is not None else z

    # ---------------------------------------------------------- categorical
    def _fit_categorical(self, df):
        self.cat_cols = [c for c in config.CATEGORICAL if c in df.columns]
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.ohe.fit(df[self.cat_cols].astype(str).fillna("missing"))
        # list-valued columns (amenities, host_verifications) -> multi-hot via dictionary
        self.list_vocab = {}
        for col in [c for c in config.LIST_COLUMNS if c in df.columns]:
            counts = {}
            for v in df[col]:
                for tok in set(_tokenize_list(v)):
                    counts[tok] = counts.get(tok, 0) + 1
            keep = sorted(t for t, n in counts.items() if n >= config.LIST_MIN_FREQ * len(df))
            self.list_vocab[col] = {t: i for i, t in enumerate(keep)}

    def _tf_categorical(self, df):
        blocks = [self.ohe.transform(df.reindex(columns=self.cat_cols).astype(str).fillna("missing"))]
        for col, vocab in self.list_vocab.items():
            m = np.zeros((len(df), len(vocab)), dtype=np.float32)
            if col in df.columns:
                for r, v in enumerate(df[col]):
                    for tok in _tokenize_list(v):
                        j = vocab.get(tok)
                        if j is not None:
                            m[r, j] = 1.0
            blocks.append(m)
        return np.hstack(blocks)

    # ----------------------------------------------------------------- text
    def _fit_text(self, df):
        self.text_models = {}
        for col in [c for c in config.TEXT_COLUMNS if c in df.columns]:
            vec = TEXT_MODES[self.text_mode]()
            mat = vec.fit_transform(df[col].fillna("").astype(str))
            k = min(config.SVD_DIM, max(1, mat.shape[1] - 1))
            svd = TruncatedSVD(n_components=k, random_state=config.SEED).fit(mat)
            self.text_models[col] = (vec, svd, k)

    def _tf_text(self, df):
        blocks = []
        for col, (vec, svd, k) in self.text_models.items():
            text = df[col].fillna("").astype(str) if col in df.columns else pd.Series([""] * len(df))
            blocks.append(svd.transform(vec.transform(text)))
        return np.hstack(blocks)

    # ----------------------------------------------------------------- date
    def _fit_date(self, df):
        self.date_cols = [c for c in config.DATE_COLUMNS if c in df.columns]
        self.date_mean, self.date_min = {}, {}
        for c in self.date_cols:
            d = pd.to_datetime(df[c], errors="coerce")
            self.date_mean[c] = d.mean()
            self.date_min[c] = d.min()
        self.date_scaler = StandardScaler().fit(self._date_matrix(df))

    def _date_matrix(self, df):
        cols = []
        for c in self.date_cols:
            d = pd.to_datetime(df[c], errors="coerce") if c in df.columns else pd.Series(pd.NaT, index=df.index)
            d = d.fillna(self.date_mean[c])                      # null -> mean date
            cols.append((d - self.date_min[c]).dt.days.astype(float).values)   # days since earliest
        return np.column_stack(cols)

    def _tf_date(self, df):
        return self.date_scaler.transform(self._date_matrix(df))
