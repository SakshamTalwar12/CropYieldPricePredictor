# Cascaded Crop Yield → Mandi Price Predictor

Full-stack implementation of the paper *"A Cascaded Machine Learning Framework for Joint Crop Yield and Mandi Price Prediction"* (Kumar & Talwar).

```
USER INPUT ─┬─ Weather / Soil ─► Yield model (Random Forest) ─► Predicted yield ─┐
            └─ Crop / Market ───────────────────────────────────────────────────┴─► Price model (XGBoost / LSTM / LR) ─► Dashboard
```

* **Backend:** Python, FastAPI, scikit-learn, XGBoost, optional PyTorch LSTM
* **Frontend:** dependency-free HTML/CSS/JS + Chart.js (vendored, works offline), served by FastAPI
* **Models are pre-trained and committed** (`models/`), so it runs right after cloning.

## Quick start (VS Code)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional LSTM price model:  pip install -r requirements-lstm.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```
Open http://localhost:8000 (API docs at `/docs`). In VS Code, press **F5** ("Run API + dashboard"); tasks and tests are in `.vscode/`.
Run tests: `python -m pytest -q`.

If the pickled models fail to load (different library versions), retrain: `python -m backend.ml.train_cascade` (~3 min).
Without PyTorch the LSTM option is simply hidden.

## Data

| Stage | Dataset | How it is made |
|---|---|---|
| Yield | Crop-yield CSV (110k district-year rows, 1966–2017, 20 states, 23 crops). Soil and weather are embedded in the text `prompt` column. | `backend/ml/prepare_yield_data.py` parses it with regexes into `data/yield_processed.csv.gz`. 76k rows have complete weather and are used. |
| Price | **Built here** (no real price file was provided) | `backend/ml/build_price_dataset.py` follows the paper's synthetic generator: crop base price × 2 %/yr trend, lowered when yield exceeds the crop's mean yield, plus noise. Added: regional factor and a shared crop-year shock. Base prices are illustrative assumptions. → `data/price_dataset.csv.gz` (37k rows, 2000–2017) |

The 86 MB raw CSV is not committed. To refresh from Hugging Face: `pip install -r requirements-hf.txt && python -m backend.ml.download_hf_dataset && python -m backend.ml.prepare_yield_data && python -m backend.ml.build_price_dataset` (that download script is untested; it could not be run from the build sandbox).

## Pipeline

```bash
python -m backend.ml.prepare_yield_data     # raw CSV -> data/yield_processed.csv.gz
python -m backend.ml.build_price_dataset    # synthetic price data
python -m backend.ml.train_cascade          # final models -> models/
python -m backend.ml.evaluate_cascade       # paper protocol -> models/metrics.json (~9 min; --fast for a smoke test)
```

Training (Algorithm 1): fit RF on weather/soil → district-grouped 5-fold **out-of-fold** yield predictions → fit price models on (crop/region/season one-hot, year, out-of-fold yield). At inference, the RF's yield goes straight into the price model.

Evaluation reproduces the paper's protocol: grouped 5-fold CV, crop-mean baselines, yield-link ablation (none / predicted / true), noise-injection test, temporal holdout (train ≤2015, test 2016–17), error by crop, feature importance, learning curve, depth sensitivity. The dashboard's **Model evidence** tab shows all of it, and forecasts display ± cross-validated RMSE bands (paper §VIII-B).

## Deviations from the paper (please read)

1. **No NDVI/humidity.** The yield data has none, so Stage 1 uses soil pH, organic carbon, texture, rainfall, temperatures and area instead (the paper lists soil as future work).
2. **Scale.** 76k rows rather than 360, so Random Forest uses `max_features=0.5` (speed and file size). Other settings follow Table A2.
3. **Region/season** are derived from state and crop (season is fixed per crop here), not chosen separately.
4. **LSTM** is a new addition (the paper did not evaluate one): it reads the last 5 years of crop-region price/yield history. It is evaluated only in the temporal holdout.
5. **Prices are synthetic**, so price metrics describe the generator, not real mandi accuracy. Swap in AGMARKNET data by replacing `data/price_dataset.csv.gz` (same columns) and retraining.

## Results on this data (from `models/metrics.json`)

| | Result |
|---|---|
| Yield, grouped CV (q/ha) | crop-mean 7.72 RMSE · LR 7.52 · RF 5.84 · **XGBoost 5.77** (R² 0.805) |
| Price, grouped CV (₹/q) | crop-mean 368.6 · LR 313.7 · RF 328.7 · **XGBoost 262.6** |
| Yield-link ablation (XGBoost) | no yield 282.1 → predicted yield 262.6 (better in all 5 folds) → true yield 256.7 |
| Noise test | benefit disappears at σ ≈ 0.9 q/ha of yield noise (the paper found ≈ 11) |
| Temporal holdout, price | LR 389.0 · RF 409.6 · **XGBoost 323.3** · LSTM 323.1 · XGBoost without yield 356.7 |

Two paper findings did **not** replicate here: the yield link still helped XGBoost under the temporal split, and Linear Regression was not the best price model. The likely reason is that this generator has a non-linear yield effect and crop-year shocks, but that was not tested. Trees also can't extrapolate beyond 2017, so for later years use Linear Regression or the LSTM (the API warns about this).

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/options` | states → districts, crops, seasons, available price models |
| `GET /api/defaults?state&district&crop` | historical weather/soil averages to pre-fill the form |
| `POST /api/predict` | run the cascade; returns yield, price, ± bands, no-yield counterfactual, charts data, warnings |
| `GET /api/metrics` | evaluation results |

## Layout

```
backend/config.py        paths, crop/season/region maps, model settings (Table A2)
backend/app/             FastAPI app, schemas, cascade inference
backend/ml/              data prep, price generator, training, evaluation, LSTM
frontend/                index.html, css/, js/app.js, vendor/chart.umd.js
data/  models/  tests/  .vscode/
```
