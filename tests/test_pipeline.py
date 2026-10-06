import numpy as np
import pandas as pd

from src import config as cfg
from src.data import time_split
from src.evaluate import fit_blend_weights, rmsle
from src.features import add_features, clean_hours, fit_hours_cap
from tests.synthetic import make_frame


def test_rmsle_zero_for_perfect_prediction():
    y = np.array([1000.0, 5000.0, 20000.0])
    assert rmsle(y, y) == 0.0


def test_sentinel_year_and_age():
    df = make_frame(200)
    out = add_features(df, pd.to_datetime(df[cfg.DATE_COL]).min())
    assert (out["ManufactureYear"] == cfg.MANUFACTURE_YEAR_SENTINEL).sum() == 0
    assert out["Age"].dropna().between(0, cfg.MAX_ASSET_AGE).all()
    assert not set(cfg.EMPTY_COLS) & set(out.columns)


def test_time_split_has_no_overlap_in_time():
    df = make_frame(500)
    out = add_features(df, pd.to_datetime(df[cfg.DATE_COL]).min())
    train, ev = time_split(out)
    assert train[cfg.DATE_COL].max() < ev[cfg.DATE_COL].min()


def test_hours_cap_uses_given_cap_and_nulls_zero():
    df = pd.DataFrame({"OperationalHoursMeter": [0.0, 10.0, 500.0, 10**7, np.nan]})
    out = clean_hours(df, cap=1000.0)
    assert out["OperationalHoursMeter"].isna().tolist() == [True, False, False, True, True]
    assert fit_hours_cap(pd.Series(range(1001)), 0.5) == 500.0


def test_blend_weights_are_valid_simplex():
    rng = np.random.default_rng(0)
    y = rng.uniform(5000, 60000, 400)
    P = np.column_stack([y * rng.normal(1, 0.05, 400), y * rng.normal(1, 0.3, 400)])
    w = fit_blend_weights(P, y)
    assert np.isclose(w.sum(), 1) and (w >= 0).all() and w[0] > w[1]
