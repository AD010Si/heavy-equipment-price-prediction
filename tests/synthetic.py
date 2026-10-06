"""Tiny synthetic dataset with the same schema as the competition data (for tests/CI)."""
import numpy as np
import pandas as pd


def make_frame(n=1500, seed=0, with_target=True):
    rng = np.random.default_rng(seed)
    year = rng.integers(1985, 2012, n)
    mfg = rng.integers(1975, 2010, n).astype(float)
    mfg[rng.random(n) < 0.1] = 1001
    hours = rng.gamma(2.0, 2500, n)
    hours[rng.random(n) < 0.4] = np.nan
    hours[rng.random(n) < 0.01] = 9e6
    dates = pd.to_datetime(year.astype(str) + "-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    pick = lambda opts, p=0.0: np.where(rng.random(n) < p, None, rng.choice(opts, n))
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "AssetID": rng.integers(0, 500, n),
        "TransactionDate": dates.strftime("%Y-%m-%d"),
        "ManufactureYear": mfg,
        "OperationalHoursMeter": hours,
        "ProductConfigID": rng.integers(0, 60, n),
        "InventoryGroupCategory": pick(list("ABCD")),
        "UtilizationTier": pick(["High", "Low", "Mid"], 0.2),
        "DataOriginCode": pick(["x", "y"]),
        "RegionCode": pick([f"R{i}" for i in range(12)]),
        "AssetScaleFactor": pick(["Small", "Large"], 0.3),
        "CabinType": pick(["EROPS", "OROPS"], 0.5),
        "DrivetrainType": pick(["2WD", "4WD"], 0.5),
        "Spec_BaseClass": pick([f"B{i}" for i in range(30)]),
        "Spec_SubClass": pick([f"S{i}" for i in range(15)], 0.6),
        "Spec_ReleaseSeries": pick([f"E{i}" for i in range(10)], 0.7),
        "Spec_VariantModifier": pick(["a", "b", "c"], 0.8),
        "FunctionalClassification": pick([f"F{i}" for i in range(20)]),
        "col4": np.nan, "col5": np.nan, "col18": np.nan, "col19": np.nan,
    })
    if with_target:
        age = (year - np.where(mfg == 1001, 1990, mfg)).clip(0, 40)
        df["TargetValue"] = np.exp(10.9 - 0.03 * age + rng.normal(0, 0.25, n))
    return df
