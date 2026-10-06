"""Train all models, evaluate on a chronological hold-out and save artifacts.

Usage:
    python -m src.train --data-dir data/raw
    python -m src.train --data-dir data/raw --gpu        # CatBoost/XGBoost on GPU
    python -m src.train --data-dir data/raw --fast       # tiny models, sanity check only

Targets are modelled as log1p(TargetValue); metric is RMSLE.

Ensemble honesty: blend weights are fitted on the *first half* of the evaluation
window and scored on the *second half*, so the reported blend score is not
computed on the same rows that were used to choose the weights. The weights
saved for the final submission are then re-fitted on the whole evaluation window.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_raw, time_split
from .evaluate import fit_blend_weights, rmsle
from .features import add_features, clean_hours, fit_hours_cap
from .models import baseline_models, final_models
from .preprocess import build_catboost_preprocessor, build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def prepare_splits(train_raw: pd.DataFrame):
    """Feature-engineer, split chronologically and clean hours using train-only stats."""
    reference_date = pd.to_datetime(train_raw[cfg.DATE_COL]).min()
    train = add_features(train_raw, reference_date)
    train_df, eval_df = time_split(train)

    hours_cap = fit_hours_cap(train_df["OperationalHoursMeter"])
    train_df = clean_hours(train_df, hours_cap)
    eval_df = clean_hours(eval_df, hours_cap)

    X_train, y_train = train_df.drop(columns=[cfg.TARGET]), train_df[cfg.TARGET]
    X_eval, y_eval = eval_df.drop(columns=[cfg.TARGET]), eval_df[cfg.TARGET]
    return X_train, y_train, X_eval, y_eval, reference_date, hours_cap


def run(data_dir, models_dir=cfg.MODELS_DIR, reports_dir=cfg.REPORTS_DIR,
        fast: bool = False, gpu: bool = False) -> dict:
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_raw, _ = load_raw(data_dir)
    X_train, y_train, X_eval, y_eval, reference_date, hours_cap = prepare_splits(train_raw)
    log.info("Train %s | Eval %s | hours cap %.0f", X_train.shape, X_eval.shape, hours_cap)

    pre, cat_pre = build_preprocessor(), build_catboost_preprocessor()
    Xtr, Xev = pre.fit_transform(X_train), pre.transform(X_eval)
    Xtr_cat, Xev_cat = cat_pre.fit_transform(X_train), cat_pre.transform(X_eval)
    y_log = np.log1p(y_train)

    # 1) Model-family comparison ------------------------------------------------
    baseline_scores = {}
    for name, model in baseline_models(fast).items():
        model.fit(Xtr, y_log)
        baseline_scores[name] = rmsle(y_eval, np.expm1(model.predict(Xev)))
        log.info("baseline %-18s RMSLE %.5f", name, baseline_scores[name])

    # 2) Final gradient-boosting models ----------------------------------------
    fitted, eval_preds = {}, {}
    for name, model in final_models(fast, gpu).items():
        if name == "CatBoost":
            model.fit(Xtr_cat, y_log, cat_features=cfg.CAT_FEATURES)
            pred = model.predict(Xev_cat)
        else:
            model.fit(Xtr, y_log)
            pred = model.predict(Xev)
        fitted[name], eval_preds[name] = model, np.clip(np.expm1(pred), 0, None)
        log.info("final    %-18s RMSLE %.5f", name, rmsle(y_eval, eval_preds[name]))

    # 3) Ensemble with an honest hold-out for the weights ----------------------
    names = list(eval_preds)
    P = np.column_stack([eval_preds[n] for n in names])
    y_eval_arr = y_eval.to_numpy()
    mid = len(y_eval_arr) // 2                      # eval is already sorted by date
    w_half = fit_blend_weights(P[:mid], y_eval_arr[:mid])
    blend_holdout = rmsle(y_eval_arr[mid:], P[mid:] @ w_half)
    single_holdout = {n: rmsle(y_eval_arr[mid:], P[mid:, i]) for i, n in enumerate(names)}

    w_full = fit_blend_weights(P, y_eval_arr)       # used for the submission
    blend_in_sample = rmsle(y_eval_arr, P @ w_full)
    log.info("blend (weights fit on 1st half) RMSLE on 2nd half: %.5f", blend_holdout)
    log.info("blend (weights fit on all eval, in-sample)       : %.5f", blend_in_sample)

    # 4) Persist ---------------------------------------------------------------
    joblib.dump(
        {
            "reference_date": reference_date,
            "hours_cap": hours_cap,
            "preprocessor": pre,
            "catboost_preprocessor": cat_pre,
            "models": fitted,
            "weights": dict(zip(names, map(float, w_full))),
        },
        models_dir / "artifacts.joblib",
    )
    metrics = {
        "baselines_eval_rmsle": baseline_scores,
        "final_models_eval_rmsle": {n: rmsle(y_eval, eval_preds[n]) for n in names},
        "final_models_second_half_rmsle": single_holdout,
        "blend_weights_fit_on_first_half": dict(zip(names, map(float, w_half))),
        "blend_rmsle_second_half_holdout": blend_holdout,
        "blend_weights_final": dict(zip(names, map(float, w_full))),
        "blend_rmsle_in_sample": blend_in_sample,
        "n_train": int(len(X_train)),
        "n_eval": int(len(X_eval)),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Saved artifacts to %s and metrics to %s", models_dir, reports_dir / "metrics.json")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=cfg.DATA_DIR)
    ap.add_argument("--gpu", action="store_true", help="use GPU for CatBoost / XGBoost")
    ap.add_argument("--fast", action="store_true", help="tiny models for a quick sanity run")
    args = ap.parse_args()
    run(args.data_dir, fast=args.fast, gpu=args.gpu)


if __name__ == "__main__":
    main()
