import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample():
    opts = client.get("/api/options").json()
    state = "Punjab" if "Punjab" in opts["states"] else sorted(opts["states"])[0]
    district, crop = opts["states"][state][0], "Wheat" if "Wheat" in opts["crops"] else opts["crops"][0]
    d = client.get("/api/defaults", params=dict(state=state, district=district, crop=crop)).json()
    body = dict(state=state, district=district, crop=crop, year=2017,
                **{k: d[k] for k in ("rainfall_mm", "temp_avg_c", "temp_min_c", "temp_max_c", "soil_ph",
                                      "organic_carbon_gkg", "soil_texture", "area_kha")})
    return opts, body


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_options(sample):
    opts, _ = sample
    assert len(opts["states"]) == 20 and "xgboost" in opts["price_models"]


@pytest.mark.parametrize("model", ["xgboost", "linear", "lstm"])
def test_predict_cascade(sample, model):
    opts, body = sample
    if model not in opts["price_models"]:
        pytest.skip("LSTM not available")
    r = client.post("/api/predict", json={**body, "price_model": model})
    assert r.status_code == 200
    j = r.json()
    assert j["yield_forecast"]["value"] > 0 and j["price_forecast"]["value"] > 0
    assert j["price_vs_yield"] and j["price_trend"]


def test_validation_errors(sample):
    _, body = sample
    assert client.post("/api/predict", json={**body, "crop": "Nope"}).status_code == 422
    assert client.post("/api/predict", json={**body, "rainfall_mm": -5}).status_code == 422
    assert client.post("/api/predict", json={**body, "temp_min_c": 30, "temp_max_c": 10}).status_code == 422


def test_future_year_warns(sample):
    _, body = sample
    j = client.post("/api/predict", json={**body, "year": 2030}).json()
    assert any("beyond the training data" in w for w in j["warnings"])


def test_frontend_served():
    assert "Crop Yield" in client.get("/").text
    assert client.get("/static/js/app.js").status_code == 200
