"""Model definitions.

Two groups:
* ``baseline_models``  - five reference models with fixed, hand-picked settings,
  used to compare model families.
* ``final_models``     - the gradient-boosting models that go into the ensemble.

CatBoost parameters come from an Optuna study (time-series CV, 25 trials). The
LightGBM / XGBoost parameters were chosen by hand and are *not* Optuna-tuned.
"""
from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import config as cfg

SEED = cfg.RANDOM_STATE


def baseline_models(fast: bool = False) -> dict:
    s = 0.1 if fast else 1.0  # shrink ensembles for quick sanity runs
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=max(10, int(200 * s)), min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ),
        "XGBoost": XGBRegressor(
            n_estimators=max(20, int(800 * s)), learning_rate=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=max(20, int(1500 * s)), learning_rate=0.03, num_leaves=63,
            random_state=SEED, verbosity=-1,
        ),
    }


def final_models(fast: bool = False, gpu: bool = False) -> dict:
    s = 0.1 if fast else 1.0
    catboost = CatBoostRegressor(
        iterations=max(30, int(1167 * s)),
        depth=9,
        learning_rate=0.0784,
        l2_leaf_reg=2.674,
        subsample=0.855,
        bootstrap_type="Bernoulli",
        task_type="GPU" if gpu else "CPU",
        random_seed=SEED,
        verbose=0,
    )
    lightgbm = LGBMRegressor(
        n_estimators=max(30, int(1250 * s)), num_leaves=63, learning_rate=0.03,
        colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    xgboost = XGBRegressor(
        n_estimators=max(30, int(1000 * s)), max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, tree_method="hist",
        device="cuda" if gpu else "cpu", random_state=SEED,
    )
    return {"CatBoost": catboost, "LightGBM": lightgbm, "XGBoost": xgboost}
