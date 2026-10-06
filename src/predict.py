"""Create a Kaggle submission from the artifacts saved by ``src.train``.

Usage:
    python -m src.predict --data-dir data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_raw
from .features import add_features, clean_hours


def make_submission(data_dir, models_dir=cfg.MODELS_DIR, out_path=None) -> pd.DataFrame:
    art = joblib.load(Path(models_dir) / "artifacts.joblib")
    _, test_raw = load_raw(data_dir)

    test = clean_hours(add_features(test_raw, art["reference_date"]), art["hours_cap"])
    X, X_cat = art["preprocessor"].transform(test), art["catboost_preprocessor"].transform(test)

    blended = np.zeros(len(test))
    for name, weight in art["weights"].items():
        if weight == 0:
            continue
        model = art["models"][name]
        raw = model.predict(X_cat if name == "CatBoost" else X)
        blended += weight * np.expm1(raw)

    submission = pd.DataFrame({cfg.ID_COL: test_raw[cfg.ID_COL], cfg.TARGET: np.clip(blended, 0, None)})
    out_path = Path(out_path or cfg.SUBMISSIONS_DIR / "submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    return submission


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=cfg.DATA_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sub = make_submission(args.data_dir, out_path=args.out)
    print(sub[cfg.TARGET].describe())


if __name__ == "__main__":
    main()
