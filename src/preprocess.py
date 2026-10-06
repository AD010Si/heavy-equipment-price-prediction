"""Scikit-learn preprocessing pipelines."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as cfg


def build_preprocessor() -> ColumnTransformer:
    """Pipeline for LightGBM / XGBoost / Random Forest / linear models.

    * numeric: median impute + standardise
    * low-cardinality categoricals: 'Unknown' + one-hot
    * high-cardinality categoricals: 'Unknown' + ordinal codes (unseen -> -1)
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    low_card = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_card = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", numeric, cfg.NUMERIC_FEATURES),
        ("low_card", low_card, cfg.LOW_CARD_CAT),
        ("high_card", high_card, cfg.HIGH_CARD_CAT),
    ]).set_output(transform="pandas")


def build_catboost_preprocessor() -> ColumnTransformer:
    """CatBoost handles categoricals natively, so only impute (no encoding)."""
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), cfg.NUMERIC_FEATURES),
            ("cat", SimpleImputer(strategy="constant", fill_value="Unknown"), cfg.CAT_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
