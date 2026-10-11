"""Leakage-aware evaluation protocol from the paper (Sections VI-VII).

 * Stage 1: district-grouped 5-fold CV for crop-mean baseline, LR, RF, XGBoost.
 * Stage 2: nested protocol - out-of-fold RF yield for training rows, RF fitted on the
   training portion for test rows. Baseline / LR / RF / XGBoost price models.
 * Ablation: no yield feature vs predicted yield vs true yield.
 * Noise injection on the true yield fed to the trained price model.
 * Temporal holdout (train <= 2015, test 2016-2017).
 * Errors by crop / region / season, feature importance, learning curve, depth sensitivity.

Writes models/metrics.json (read by the API and shown in the dashboard).

Usage:  python -m backend.ml.evaluate_cascade [--fast]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from backend.config import (MODEL_DIR, RF_PRICE, RF_YIELD, SEED, TEMPORAL_TEST_YEARS,
                            XGB_PRICE, XGB_YIELD)
from backend.ml.lstm_price import TORCH_OK, LSTMPrice
from backend.ml.features import (load_price_data, load_yield_data, metrics,
                                 stage1_matrix, stage2_matrix)

N_JOBS = 2


def rf(params, **kw):
    return RandomForestRegressor(**params, n_jobs=N_JOBS, random_state=SEED, **kw)


def xg(params, **kw):
    return xgb.XGBRegressor(**params, n_jobs=N_JOBS, random_state=SEED, **kw)


def agg(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=1))} for k in df}


class CropMean:
    """Baseline: predict the mean target of each crop in the training data."""
    def fit(self, df, y):
        self.m = pd.Series(np.asarray(y), index=df["crop"].values).groupby(level=0).mean()
        self.g = float(np.mean(y)); return self
    def predict(self, df):
        return df["crop"].map(self.m).fillna(self.g).values


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- Stage 1
def stage1_cv(df, rf_params, xgb_params, k=5):
    X, y, groups = stage1_matrix(df), df["yield_qha"].values, df["district"].values
    models = {"Crop-mean baseline": None, "Linear Regression": LinearRegression,
              "Random Forest": lambda: rf(rf_params), "XGBoost": lambda: xg(xgb_params)}
    res = {m: [] for m in models}
    oof = {m: np.zeros(len(df)) for m in models}
    for f, (tr, te) in enumerate(GroupKFold(k).split(X, y, groups), 1):
        for name, make in models.items():
            if make is None:
                p = CropMean().fit(df.iloc[tr], y[tr]).predict(df.iloc[te])
            else:
                p = make().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            res[name].append(metrics(y[te], p)); oof[name][te] = p
        log(f"stage1 fold {f}/{k}")
    return {m: agg(r) for m, r in res.items()}, oof


# --------------------------------------------------------------------- Stage 2
def handoff_oof(df_tr, rf_params, k=5):
    """Out-of-fold RF yield predictions for training rows (leakage control)."""
    X, y, g = stage1_matrix(df_tr), df_tr["yield_qha"].values, df_tr["district"].values
    oof = np.zeros(len(df_tr))
    for tr, te in GroupKFold(k).split(X, y, g):
        oof[te] = rf(rf_params).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return oof


def price_models(price_rf, price_xgb):
    return {"Linear Regression": LinearRegression, "Random Forest": lambda: rf(price_rf),
            "XGBoost": lambda: xg(price_xgb)}


def run_split(tr_df, te_df, cfg, noise_sigmas=(), reps=5, rng=None, lstm=False):
    """Train the cascade on tr_df and evaluate on te_df. Returns per-model metrics."""
    out = {"yield": {}, "price": {}}
    X1_tr, X1_te = stage1_matrix(tr_df), stage1_matrix(te_df, None)
    X1_te = X1_te.reindex(columns=X1_tr.columns, fill_value=0.0)
    y_tr, y_te = tr_df["yield_qha"].values, te_df["yield_qha"].values
    p_tr, p_te = tr_df["price_rs_per_quintal"].values, te_df["price_rs_per_quintal"].values

    for name, make in {"Linear Regression": LinearRegression,
                       "Random Forest": lambda: rf(cfg["rf_y"]),
                       "XGBoost": lambda: xg(cfg["xgb_y"])}.items():
        out["yield"][name] = metrics(y_te, make().fit(X1_tr, y_tr).predict(X1_te))

    oof = handoff_oof(tr_df, cfg["rf_y"])
    yhat_te = rf(cfg["rf_y"]).fit(X1_tr, y_tr).predict(X1_te)
    out["handoff_yield_rmse"] = metrics(y_te, yhat_te)["rmse"]

    out["price"]["Crop-mean baseline"] = metrics(
        p_te, CropMean().fit(tr_df, p_tr).predict(te_df))
    X2_tr = stage2_matrix(tr_df, oof)
    X2_te = stage2_matrix(te_df, yhat_te, columns=list(X2_tr.columns))
    for name, make in price_models(cfg["rf_p"], cfg["xgb_p"]).items():
        out["price"][name] = metrics(p_te, make().fit(X2_tr, p_tr).predict(X2_te))

    # ablation with the XGBoost price model
    X2n_tr = stage2_matrix(tr_df, with_yield=False)
    X2n_te = stage2_matrix(te_df, with_yield=False, columns=list(X2n_tr.columns))
    out["price"]["XGBoost, no yield"] = metrics(
        p_te, xg(cfg["xgb_p"]).fit(X2n_tr, p_tr).predict(X2n_te))
    m_x = xg(cfg["xgb_p"]).fit(X2_tr, p_tr)
    out["price"]["XGBoost, true yield"] = metrics(
        p_te, m_x.predict(stage2_matrix(te_df, y_te, columns=list(X2_tr.columns))))

    if lstm and TORCH_OK:        # LSTM only makes sense with a temporal split
        pred = LSTMPrice().fit(tr_df, oof).predict(te_df, yhat_te)
        ok = ~np.isnan(pred)
        out["price"]["LSTM"] = {**metrics(p_te[ok], pred[ok]), "coverage": float(ok.mean())}

    out["noise"] = {}
    for s in noise_sigmas:
        r = [metrics(p_te, m_x.predict(stage2_matrix(
            te_df, np.clip(y_te + rng.normal(0, s, len(y_te)), 0, None),
            columns=list(X2_tr.columns))))["rmse"] for _ in range(reps)]
        out["noise"][s] = float(np.mean(r))
    return out


def grouped_price_cv(pdf, cfg, k=5, sigmas=(0, 2, 4, 6, 8, 12, 16)):
    rng = np.random.default_rng(SEED)
    folds, noise = [], []
    for f, (tr, te) in enumerate(GroupKFold(k).split(pdf, groups=pdf["district"]), 1):
        r = run_split(pdf.iloc[tr], pdf.iloc[te], cfg, sigmas, rng=rng)
        folds.append(r); log(f"price protocol fold {f}/{k}")
    out = {"yield": {}, "price": {}, "per_fold": []}
    for stage in ("yield", "price"):
        for m in folds[0][stage]:
            out[stage][m] = agg([fo[stage][m] for fo in folds])
    out["noise_rmse"] = {str(s): float(np.mean([fo["noise"][s] for fo in folds])) for s in sigmas}
    for i, fo in enumerate(folds, 1):
        out["per_fold"].append({
            "fold": i, "yield_rmse_xgb": fo["yield"]["XGBoost"]["rmse"],
            "yield_r2_xgb": fo["yield"]["XGBoost"]["r2"],
            "price_rmse_with_yield": fo["price"]["XGBoost"]["rmse"],
            "price_rmse_no_yield": fo["price"]["XGBoost, no yield"]["rmse"],
            "price_rmse_lr": fo["price"]["Linear Regression"]["rmse"]})
    nr = np.array(list(out["noise_rmse"].values()))
    ny = out["price"]["XGBoost, no yield"]["rmse"]["mean"]
    sig = np.array(sigmas, float)
    above = np.where(nr >= ny)[0]
    out["noise_breakeven_sigma"] = (float(np.interp(ny, nr[above[0] - 1:above[0] + 1],
                                                   sig[above[0] - 1:above[0] + 1]))
                                    if len(above) and above[0] > 0 else None)
    return out


# ------------------------------------------------------------------- Extras
def importance_groups(model, columns):
    gain = model.get_booster().get_score(importance_type="total_gain")
    tot = sum(gain.values())
    groups = {}
    for i, col in enumerate(columns):
        v = gain.get(f"f{i}", gain.get(col, 0.0)) / tot
        key = col.split("_")[0] if col.startswith(("crop_", "region_", "season_")) else \
            ("soil_texture" if col.startswith("soil_texture_") else col)
        groups[key] = groups.get(key, 0.0) + v
    return dict(sorted(groups.items(), key=lambda kv: -kv[1]))


def breakdown(df, oof_yield):
    d = df.assign(err2=(df["yield_qha"] - oof_yield) ** 2)
    out = {}
    for col in ("crop", "region", "season"):
        g = d.groupby(col).agg(n=("err2", "size"), mean_yield=("yield_qha", "mean"),
                               err2=("err2", "mean"))
        out[col] = [{"name": i, "n": int(r.n), "mean_yield": float(r.mean_yield),
                     "yield_rmse": float(np.sqrt(r.err2))} for i, r in g.iterrows()]
    return out


def learning_curve(df, xgb_params, k=5):
    X, y, g = stage1_matrix(df), df["yield_qha"].values, df["district"].values
    rng = np.random.default_rng(SEED)
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        scores = []
        for tr, te in GroupKFold(k).split(X, y, g):
            tr_groups = np.unique(g[tr])
            keep = rng.choice(tr_groups, max(1, int(len(tr_groups) * frac)), replace=False)
            tr2 = tr[np.isin(g[tr], keep)]
            scores.append(metrics(y[te], xg(xgb_params).fit(X.iloc[tr2], y[tr2]).predict(X.iloc[te])))
        rows.append({"fraction": frac, "rmse": float(np.mean([s["rmse"] for s in scores])),
                     "r2": float(np.mean([s["r2"] for s in scores]))})
    return rows


def depth_sensitivity(df, xgb_params, k=5):
    X, y, g = stage1_matrix(df), df["yield_qha"].values, df["district"].values
    rows = []
    for d in (2, 3, 5, 7):
        s = [metrics(y[te], xg({**xgb_params, "max_depth": d}).fit(X.iloc[tr], y[tr])
                     .predict(X.iloc[te]))["rmse"] for tr, te in GroupKFold(k).split(X, y, g)]
        rows.append({"max_depth": d, "rmse_mean": float(np.mean(s)), "rmse_std": float(np.std(s, ddof=1))})
    return rows


def main(fast=False):
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    cfg = dict(rf_y=dict(RF_YIELD), xgb_y=dict(XGB_YIELD), rf_p=dict(RF_PRICE), xgb_p=dict(XGB_PRICE))
    ydf, pdf = load_yield_data(), load_price_data()
    if fast:
        cfg["rf_y"]["n_estimators"] = cfg["rf_p"]["n_estimators"] = 60
        ydf = ydf.sample(frac=0.4, random_state=SEED).reset_index(drop=True)
        pdf = pdf.sample(frac=0.4, random_state=SEED).reset_index(drop=True)
    R = {"config": {"fast": fast, "n_yield_rows": len(ydf), "n_price_rows": len(pdf),
                    "protocol": "district-grouped 5-fold CV", **{k: v for k, v in cfg.items()}}}

    log("Stage 1 grouped CV (full yield dataset)")
    R["stage1_cv"], oof = stage1_cv(ydf, cfg["rf_y"], cfg["xgb_y"])
    R["breakdown"] = breakdown(ydf, oof["XGBoost"])
    X1 = stage1_matrix(ydf)
    R["feature_importance"] = importance_groups(xg(cfg["xgb_y"]).fit(X1, ydf["yield_qha"]), list(X1.columns))
    log("learning curve / sensitivity")
    R["learning_curve"] = learning_curve(ydf, cfg["xgb_y"])
    R["depth_sensitivity"] = depth_sensitivity(ydf, cfg["xgb_y"])

    log("Cascade protocol, grouped CV (price dataset)")
    R["cascade_cv"] = grouped_price_cv(pdf, cfg)

    log("Temporal holdout")
    lo, hi = TEMPORAL_TEST_YEARS
    tr, te = pdf[pdf["year"] < lo], pdf[(pdf["year"] >= lo) & (pdf["year"] <= hi)]
    R["temporal"] = run_split(tr, te, cfg, lstm=True)
    R["temporal"]["train_years"] = f"{int(tr.year.min())}-{int(tr.year.max())}"
    R["temporal"]["test_years"] = f"{lo}-{hi}"

    # Error bands shown next to each forecast in the dashboard (paper Sec. VIII-B)
    R["error_bands"] = {
        "yield_rmse_qha": R["stage1_cv"]["Random Forest"]["rmse"]["mean"],
        "yield_rmse_qha_xgb": R["stage1_cv"]["XGBoost"]["rmse"]["mean"],
        "price_rmse_xgb": R["cascade_cv"]["price"]["XGBoost"]["rmse"]["mean"],
        "price_rmse_lr": R["cascade_cv"]["price"]["Linear Regression"]["rmse"]["mean"],
        "price_rmse_xgb_temporal": R["temporal"]["price"]["XGBoost"]["rmse"],
        "price_rmse_lr_temporal": R["temporal"]["price"]["Linear Regression"]["rmse"],
    }
    R["runtime_seconds"] = round(time.time() - t0)
    path = MODEL_DIR / ("metrics_fast.json" if fast else "metrics.json")
    path.write_text(json.dumps(R, indent=2))
    log(f"wrote {path} in {R['runtime_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="40%% subsample, 60 trees (smoke test)")
    main(ap.parse_args().fast)
