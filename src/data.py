"""Data loading and the chronological train / evaluation split."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as cfg


def load_raw(data_dir: str | Path = cfg.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. See data/README.md for download instructions."
            )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def time_split(
    df: pd.DataFrame, cutoff_quantile: float = cfg.EVAL_FRACTION_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the past, evaluate on the most recent transactions.

    A random split would let the model see the future; sales prices drift over time,
    so a chronological hold-out is a more realistic estimate of deployment error.
    """
    ordered = df.sort_values(cfg.DATE_COL).reset_index(drop=True)
    cutoff = ordered[cfg.DATE_COL].quantile(cutoff_quantile)
    return (
        ordered[ordered[cfg.DATE_COL] < cutoff].copy(),
        ordered[ordered[cfg.DATE_COL] >= cutoff].copy(),
    )
