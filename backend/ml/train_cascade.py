"""Train and save the deployed cascade (Algorithm 1 of the paper).

 1. Fit Stage-1 yield models on all weather+soil rows (RF = hand-off, XGBoost, LR).
 2. Obtain 5-fold district-grouped OUT-OF-FOLD RF yield predictions for every row.
 3. Fit Stage-2 price models on (crop/region/season one-hot, year, y_oof).
 4. Save models + lookup tables (district profiles, history series) to models/.

Usage:  python -m backend.ml.train_cascade
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from backend.config import (MODEL_DIR, REGION, RF_YIELD, SEASON, SEED, XGB_PRICE,
                            XGB_YIELD, YIELD_NUMERIC)
from backend.ml.features import (load_price_data, load_yield_data, stage1_matrix,
                                 stage2_matrix)
from backend.ml.lstm_price import TORCH_OK, LSTMPrice


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_lookups(ydf: pd.DataFrame, pdf: pd.DataFrame) -> dict:
    """Defaults and history series that the dashboard needs."""
    num = YIELD_NUMERIC
    tex = lambda s: s.mode().iat[0]
    prof = lambda g: {**{c: round(float(g[c].mean()), 2) for c in num},
                      "soil_texture": tex(g["soil_texture"]), "n": int(len(g))}
    lk = {"states": {}, "profiles": {"district_crop": {}, "district": {}, "state_crop": {},
                                      "state": {}}, "hist": {}}
    for st, g in ydf.groupby("state"):
        lk["states"][st] = sorted(g["district"].unique())
        lk["profiles"]["state"][st] = prof(g)
    for (st, c), g in ydf.groupby(["state", "crop"]):
        lk["profiles"]["state_crop"][f"{st}|{c}"] = prof(g)
    for (st, d), g in ydf.groupby(["state", "district"]):
        lk["profiles"]["district"][f"{st}|{d}"] = prof(g)
    for (st, d, c), g in ydf.groupby(["state", "district", "crop"]):
        lk["profiles"]["district_crop"][f"{st}|{d}|{c}"] = prof(g)
        yr = g.groupby("year").agg(y=("yield_qha", "mean"), r=("rainfall_mm", "mean"),
                                   t=("temp_avg_c", "mean")).round(2).reset_index()
        lk["hist"][f"{st}|{d}|{c}"] = {"year": yr["year"].astype(int).tolist(),
                                       "yield": yr["y"].tolist(), "rainfall": yr["r"].tolist(),
                                       "temp": yr["t"].tolist()}
    pr = pdf.groupby(["crop", "region", "year"])["price_rs_per_quintal"].mean().round(1).reset_index()
    lk["price_hist"] = {f"{c}|{r}": {"year": g["year"].astype(int).tolist(),
                                     "price": g["price_rs_per_quintal"].tolist()}
                        for (c, r), g in pr.groupby(["crop", "region"])}
    lk["crops"] = sorted(ydf["crop"].unique())
    lk["crop_season"] = {c: SEASON[c] for c in lk["crops"]}
    lk["state_region"] = {s: REGION[s] for s in lk["states"]}
    lk["crop_mean_yield"] = ydf.groupby("crop")["yield_qha"].mean().round(2).to_dict()
    lk["max_year"] = int(max(ydf["year"].max(), pdf["year"].max()))
    return lk


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    ydf, pdf = load_yield_data(), load_price_data()
    X1 = stage1_matrix(ydf); cols1 = list(X1.columns); y = ydf["yield_qha"].values
    mk_rf = lambda: RandomForestRegressor(**RF_YIELD, n_jobs=2, random_state=SEED)

    log("OOF hand-off predictions (district-grouped, 5-fold)")
    oof = np.zeros(len(ydf))
    for tr, te in GroupKFold(5).split(X1, y, ydf["district"]):
        oof[te] = mk_rf().fit(X1.iloc[tr], y[tr]).predict(X1.iloc[te])

    log("Final Stage-1 models")
    yield_rf = mk_rf().fit(X1, y)
    yield_xgb = xgb.XGBRegressor(**XGB_YIELD, n_jobs=2, random_state=SEED).fit(X1, y)
    yield_lr = LinearRegression().fit(X1, y)

    log("Stage-2 price models on out-of-fold yield")
    oof_by_id = pd.Series(oof, index=ydf["row_id"].values)
    pdf = pdf[pdf["row_id"].isin(oof_by_id.index)].reset_index(drop=True)
    y_oof = oof_by_id.loc[pdf["row_id"]].values
    X2 = stage2_matrix(pdf, y_oof); cols2 = list(X2.columns)
    p = pdf["price_rs_per_quintal"].values
    price_xgb = xgb.XGBRegressor(**XGB_PRICE, n_jobs=2, random_state=SEED).fit(X2, p)
    price_lr = LinearRegression().fit(X2, p)
    X2n = stage2_matrix(pdf, with_yield=False)
    price_xgb_noyield = xgb.XGBRegressor(**XGB_PRICE, n_jobs=2, random_state=SEED).fit(X2n, p)

    bundle = {"yield_rf": yield_rf, "yield_xgb": yield_xgb, "yield_lr": yield_lr,
              "price_xgb": price_xgb, "price_lr": price_lr, "price_xgb_noyield": price_xgb_noyield,
              "cols1": cols1, "cols2": cols2, "cols2_noyield": list(X2n.columns)}
    joblib.dump(bundle, MODEL_DIR / "cascade.joblib", compress=3)

    if TORCH_OK:
        import torch
        log("LSTM price model")
        lstm = LSTMPrice().fit(pdf, y_oof)
        torch.save(lstm.state(), MODEL_DIR / "lstm_price.pt")
    else:
        log("PyTorch not installed - skipping LSTM (optional)")

    (MODEL_DIR / "lookups.json").write_text(json.dumps(build_lookups(ydf, pdf)))
    log(f"done in {time.time() - t0:.0f}s -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
