from typing import Literal, Optional

from pydantic import BaseModel, Field

PriceModelName = Literal["xgboost", "linear", "lstm"]


class PredictRequest(BaseModel):
    state: str
    district: str
    crop: str
    year: int = Field(2017, ge=2000, le=2040)
    # Weather / soil inputs (Stage 1)
    rainfall_mm: float = Field(..., ge=0, le=6000)
    temp_avg_c: float = Field(..., ge=-10, le=45)
    temp_min_c: float = Field(..., ge=-40, le=40)
    temp_max_c: float = Field(..., ge=0, le=60)
    soil_ph: float = Field(..., ge=3, le=10)
    organic_carbon_gkg: float = Field(..., ge=0, le=100)
    soil_texture: Literal["loamy", "silty", "clay-heavy"] = "loamy"
    area_kha: float = Field(..., ge=0, le=3000, description="Area, thousand hectares")
    price_model: PriceModelName = "xgboost"


class Forecast(BaseModel):
    value: float
    unit: str
    low: Optional[float] = None
    high: Optional[float] = None
    model: str
    band_note: Optional[str] = None


class PredictResponse(BaseModel):
    region: str
    season: str
    yield_forecast: Forecast
    yield_xgboost_qha: float
    price_forecast: Forecast
    price_without_yield: float
    cascade_effect_rs: float
    revenue_per_ha: float
    price_vs_yield: list[dict]
    price_trend: list[dict]
    district_history: Optional[dict] = None
    price_history: Optional[dict] = None
    warnings: list[str]
