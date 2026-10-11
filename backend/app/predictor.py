"""Cascade inference: weather/soil -> Yield (RF) -> predicted yield -> Price model."""
from __future__ import annotations

import json
import math
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from backend.app.schemas import Forecast, PredictRequest, PredictResponse
from backend.config import MODEL_DIR, REGION, SEASON, YIELD_NUMERIC
from backend.ml.features import stage1_matrix, stage2_matrix
from backend.ml.lstm_price import TORCH_OK, LSTMPrice


class ModelsNotTrained(RuntimeError):
    pass


class Predictor:
    def __init__(self):
        path = MODEL_DIR / "cascade.joblib"
        if not path.exists() or not (MODEL_DIR / "lookups.json").exists():
            raise ModelsNotTrained(
                "Trained models not found. Run `python -m backend.ml.train_cascade`.")
        try:
            self.m = joblib.load(path)
        except Exception as e:  # version mismatch between pickle and installed libs
            raise ModelsNotTrained(
                f"Could not load models/cascade.joblib ({type(e).__name__}). Retrain with "
                "`python -m backend.ml.train_cascade` using your installed library versions.")
        self.lookups = json.loads((MODEL_DIR / "lookups.json").read_text())
        self.lstm = None
        lp = MODEL_DIR / "lstm_price.pt"
        if TORCH_OK and lp.exists():
            import torch
            self.lstm = LSTMPrice.from_state(torch.load(lp, weights_only=False))

    # ------------------------------------------------------------------ meta
    @cached_property
    def metrics(self) -> dict | None:
        p = MODEL_DIR / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    def options(self) -> dict:
        lk = self.lookups
        return {"states": lk["states"], "crops": lk["crops"], "crop_season": lk["crop_season"],
                "state_region": lk["state_region"], "max_year": lk["max_year"],
                "price_models": ["xgboost", "linear"] + (["lstm"] if self.lstm else [])}

    def defaults(self, state: str, district: str, crop: str) -> dict:
        P = self.lookups["profiles"]
        for level, key in (("district_crop", f"{state}|{district}|{crop}"),
                           ("state_crop", f"{state}|{crop}"),
                           ("district", f"{state}|{district}"), ("state", state)):
            if key in P[level]:
                return {"level": level, **P[level][key]}
        raise KeyError("unknown state")

    # ------------------------------------------------------------- prediction
    def _bands(self):
        e = (self.metrics or {}).get("error_bands", {})
        return e

    def predict(self, r: PredictRequest) -> PredictResponse:
        lk = self.lookups
        if r.state not in lk["states"]:
            raise ValueError(f"Unknown state: {r.state}")
        if r.district not in lk["states"][r.state]:
            raise ValueError(f"Unknown district for {r.state}: {r.district}")
        if r.crop not in lk["crops"]:
            raise ValueError(f"Unknown crop: {r.crop}")
        if r.temp_min_c > r.temp_max_c:
            raise ValueError("temp_min_c must not exceed temp_max_c")

        region, season = REGION[r.state], SEASON[r.crop]
        row = pd.DataFrame([{**r.model_dump(), "region": region, "season": season,
                             "temp_range_c": r.temp_max_c - r.temp_min_c}])
        warns: list[str] = []
        bands = self._bands()

        # ---- Stage 1: yield (Random Forest is the hand-off model, as in the paper)
        X1 = stage1_matrix(row, self.m["cols1"])
        y_rf = float(self.m["yield_rf"].predict(X1)[0])
        y_xgb = float(self.m["yield_xgb"].predict(X1)[0])
        y_rf = max(y_rf, 0.0)
        yb = bands.get("yield_rmse_qha")
        yf = Forecast(value=y_rf, unit="quintal/ha", model="Random Forest",
                      low=max(y_rf - yb, 0) if yb else None, high=y_rf + yb if yb else None,
                      band_note=f"± cross-validated RMSE ({yb:.1f} q/ha)" if yb else None)

        # ---- Stage 2: price from crop/market features + predicted yield
        def price_for(model: str, df: pd.DataFrame, yhat) -> np.ndarray:
            if model == "lstm":
                return self.lstm.predict(df, yhat)
            X2 = stage2_matrix(df, yhat, columns=self.m["cols2"])
            return self.m["price_xgb" if model == "xgboost" else "price_lr"].predict(X2)

        model = r.price_model
        if model == "lstm" and not self.lstm:
            warns.append("LSTM is unavailable (PyTorch not installed or model not trained); "
                         "using XGBoost instead.")
            model = "xgboost"
        price = float(price_for(model, row, [y_rf])[0])
        if math.isnan(price):
            warns.append(f"The LSTM has no price history for {r.crop} in the {region} region; "
                         "using XGBoost instead.")
            model = "xgboost"
            price = float(price_for(model, row, [y_rf])[0])
        names = {"xgboost": "XGBoost", "linear": "Linear Regression", "lstm": "LSTM"}
        pb = {"xgboost": bands.get("price_rmse_xgb"), "linear": bands.get("price_rmse_lr"),
              "lstm": (self.metrics or {}).get("temporal", {}).get("price", {}).get("LSTM", {}).get("rmse")}[model]
        pf = Forecast(value=price, unit="₹/quintal", model=names[model],
                      low=max(price - pb, 0) if pb else None, high=price + pb if pb else None,
                      band_note=(f"± cross-validated RMSE (₹{pb:.0f}/q)" if model != "lstm"
                                 else f"± temporal-holdout RMSE (₹{pb:.0f}/q)") if pb else None)

        X2n = stage2_matrix(row, with_yield=False, columns=self.m["cols2_noyield"])
        price_noy = float(self.m["price_xgb_noyield"].predict(X2n)[0])

        # ---- Sensitivity of price to yield (the cascade link) and price trend over years
        mult = np.linspace(0.6, 1.4, 9)
        ys = y_rf * mult
        rep = pd.concat([row] * len(ys), ignore_index=True)
        p_curve = price_for(model, rep, ys)
        price_vs_yield = [{"yield": round(float(a), 2), "price": round(float(b), 1)}
                          for a, b in zip(ys, p_curve) if not math.isnan(b)]
        years = list(range(2000, max(r.year, lk["max_year"]) + 1))
        rep = pd.concat([row] * len(years), ignore_index=True)
        rep["year"] = years
        p_tr = price_for(model, rep, [y_rf] * len(years))
        price_trend = [{"year": y, "price": round(float(p), 1)} for y, p in zip(years, p_tr)
                       if not math.isnan(p)]

        # ---- Warnings
        if r.year > lk["max_year"]:
            warns.append(f"Year {r.year} is beyond the training data (≤ {lk['max_year']}). Tree "
                         "models cannot extrapolate a price trend; Linear Regression or the LSTM "
                         "is more appropriate (paper §VII-E).")
        prof = self.defaults(r.state, r.district, r.crop)
        if prof["level"] in ("state_crop", "state", "district"):
            warns.append(f"No data for {r.crop} in {r.district}; this combination is an extrapolation.")
        pr = self.lookups["profiles"]["state"][r.state]
        if r.rainfall_mm > 3500 or r.rainfall_mm < 50:
            warns.append("Rainfall is outside the range seen for most training records.")
        warns.append("Prices come from a synthetic generator (paper §IV-B), not market data; "
                     "treat them as a demonstration of the architecture.")

        hist = self.lookups["hist"].get(f"{r.state}|{r.district}|{r.crop}")
        ph = self.lookups["price_hist"].get(f"{r.crop}|{region}")
        return PredictResponse(
            region=region, season=season, yield_forecast=yf, yield_xgboost_qha=max(y_xgb, 0.0),
            price_forecast=pf, price_without_yield=price_noy, cascade_effect_rs=price - price_noy,
            revenue_per_ha=y_rf * price, price_vs_yield=price_vs_yield, price_trend=price_trend,
            district_history=hist, price_history=ph, warnings=warns)
