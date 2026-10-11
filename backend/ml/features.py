"""Feature construction shared by training, evaluation and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import (PRICE_NUMERIC, PRICE_DATA, YIELD_CATEGORICAL,
                            YIELD_DATA, YIELD_NUMERIC)

PRICE_CATEGORICAL = ["crop", "region", "season"]
STAGE1_COLS = YIELD_NUMERIC + YIELD_CATEGORICAL


def load_yield_data() -> pd.DataFrame:
    """Rows with complete weather + soil, with gross outliers removed
    (yield > 5x the crop median, which are data-entry artefacts)."""
    df = pd.read_csv(YIELD_DATA).dropna(subset=STAGE1_COLS + ["yield_qha"])
    med = df.groupby("crop")["yield_qha"].transform("median")
    df = df[df["yield_qha"] <= 5 * med]
    df = df.reset_index(drop=True)
    df["row_id"] = df.index          # stable id shared with the price dataset
    return df


def load_price_data() -> pd.DataFrame:
    return pd.read_csv(PRICE_DATA)


def one_hot(df: pd.DataFrame, numeric: list[str], categorical: list[str],
            columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode; if `columns` is given, align to exactly that column list."""
    X = pd.concat([df[numeric].astype(float),
                   pd.get_dummies(df[categorical], dtype=float)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def stage1_matrix(df, columns=None):
    return one_hot(df, YIELD_NUMERIC, YIELD_CATEGORICAL, columns)


def stage2_matrix(df, yhat=None, columns=None, with_yield=True):
    """Stage 2 inputs (paper Table I): one-hot crop/region/season, year, predicted yield."""
    X = one_hot(df, PRICE_NUMERIC, PRICE_CATEGORICAL)
    if with_yield:
        X["pred_yield"] = np.asarray(yhat, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"rmse": rmse, "mae": mae, "r2": r2}
