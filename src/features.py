"""Feature engineering and data-cleaning helpers.

Everything that learns a statistic from data (e.g. the hours cap) is exposed as a
``fit_*`` function so it can be fitted on the training split only and re-applied
to validation / test data without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def _stringify(value):
    return np.nan if pd.isna(value) else str(value)


def add_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Clean raw columns and derive date / age features. Returns a copy."""
    d = df.copy()

    # ManufactureYear: sentinel -> NaN, then asset age at the time of sale.
    d["ManufactureYear"] = d["ManufactureYear"].replace(cfg.MANUFACTURE_YEAR_SENTINEL, np.nan)

    d[cfg.DATE_COL] = pd.to_datetime(d[cfg.DATE_COL])
    d["sale_year"] = d[cfg.DATE_COL].dt.year
    d["sale_month"] = d[cfg.DATE_COL].dt.month
    d["sale_quarter"] = d[cfg.DATE_COL].dt.quarter
    d["sale_dow"] = d[cfg.DATE_COL].dt.dayofweek
    d["sale_elapsed"] = (d[cfg.DATE_COL] - reference_date).dt.days

    d["Age"] = d["sale_year"] - d["ManufactureYear"]
    d.loc[(d["Age"] < 0) | (d["Age"] > cfg.MAX_ASSET_AGE), "Age"] = np.nan

    # Categorical columns -> consistent object/str dtype (keeps NaN as NaN).
    for col in cfg.CAT_FEATURES:
        if col in d.columns:
            d[col] = d[col].astype(object).map(_stringify)

    return d.drop(columns=cfg.EMPTY_COLS, errors="ignore")


def fit_hours_cap(hours: pd.Series, quantile: float = cfg.HOURS_CAP_QUANTILE) -> float:
    """Upper bound for plausible operational hours, learned from training data only."""
    return float(hours.quantile(quantile))


def clean_hours(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Null out implausible meter readings (> cap or exactly 0) and add a log feature."""
    d = df.copy()
    hours = d["OperationalHoursMeter"]
    d["OperationalHoursMeter"] = hours.mask((hours > cap) | (hours == 0))
    d["OperationalHoursMeter_log"] = np.log1p(d["OperationalHoursMeter"])
    return d
