"""FastAPI app: JSON API under /api and the dashboard served from /."""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.predictor import ModelsNotTrained, Predictor
from backend.app.schemas import PredictRequest, PredictResponse
from backend.config import FRONTEND_DIR

app = FastAPI(title="Cascaded Crop Yield & Mandi Price Predictor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor()
        except ModelsNotTrained as e:
            raise HTTPException(503, str(e))
    return _predictor


@app.get("/api/health")
def health():
    try:
        p = get_predictor()
        return {"status": "ok", "lstm": p.lstm is not None, "metrics": p.metrics is not None}
    except HTTPException as e:
        return {"status": "models_missing", "detail": e.detail}


@app.get("/api/options")
def options():
    return get_predictor().options()


@app.get("/api/defaults")
def defaults(state: str = Query(...), district: str = Query(...), crop: str = Query(...)):
    try:
        return get_predictor().defaults(state, district, crop)
    except KeyError:
        raise HTTPException(404, "Unknown state")


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return get_predictor().predict(req)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/metrics")
def metrics():
    m = get_predictor().metrics
    if m is None:
        raise HTTPException(404, "metrics.json not found. Run `python -m backend.ml.evaluate_cascade`.")
    return m


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
