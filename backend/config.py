"""Central configuration: paths, crop/season/region metadata and model settings
(model settings follow Table A2 of the paper)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"

YIELD_RAW_CSV = DATA_DIR / "yield_raw.csv"          # optional: the full 86 MB HF/CSV file
YIELD_DATA = DATA_DIR / "yield_processed.csv.gz"    # parsed, compact (shipped in repo)
PRICE_DATA = DATA_DIR / "price_dataset.csv.gz"      # synthetic price dataset (built here)

SEED = 42

# ---- Table A2 of the paper -------------------------------------------------
# max_features=0.5 is a speed/size deviation from the paper (default 1.0): the real
# dataset has ~76k rows instead of 360, so a full-width RF is slow and very large.
RF_YIELD = dict(n_estimators=300, max_depth=12, min_samples_leaf=2, max_features=0.5)
XGB_YIELD = dict(n_estimators=300, max_depth=5, learning_rate=0.05)
RF_PRICE = dict(n_estimators=300, max_depth=10, max_features=0.5)
XGB_PRICE = dict(n_estimators=300, max_depth=4, learning_rate=0.05)

# Stage 1 numeric inputs. The paper used NDVI + humidity; the real dataset has no
# satellite/humidity columns, so soil variables (listed as future work) are used.
YIELD_NUMERIC = [
    "rainfall_mm", "temp_avg_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "soil_ph", "organic_carbon_gkg", "area_kha",
]
YIELD_CATEGORICAL = ["crop", "region", "season", "soil_texture"]
# Stage 2 (paper Table I): one-hot crop/region/season + year + predicted yield.
PRICE_NUMERIC = ["year"]

# ---- Metadata --------------------------------------------------------------
REGION = {
    "Uttar Pradesh": "North", "Rajasthan": "North", "Himachal Pradesh": "North",
    "Haryana": "North", "Punjab": "North", "Uttarakhand": "North",
    "Madhya Pradesh": "Central", "Chhattisgarh": "Central",
    "Gujarat": "West", "Maharashtra": "West",
    "Tamil Nadu": "South", "Karnataka": "South", "Andhra Pradesh": "South",
    "Telangana": "South", "Kerala": "South",
    "Bihar": "East", "West Bengal": "East", "Jharkhand": "East", "Orissa": "East",
    "Assam": "Northeast",
}

SEASON = {
    "Rice": "Kharif", "Maize": "Kharif", "Cotton": "Kharif", "Groundnut": "Kharif",
    "Pearl Millet": "Kharif", "Finger Millet": "Kharif", "Kharif Sorghum": "Kharif",
    "Sorghum": "Kharif", "Pigeonpea": "Kharif", "Soyabean": "Kharif",
    "Sesamum": "Kharif", "Castor": "Kharif", "Oilseeds": "Kharif",
    "Wheat": "Rabi", "Barley": "Rabi", "Chickpea": "Rabi", "Linseed": "Rabi",
    "Rapeseed And Mustard": "Rabi", "Rabi Sorghum": "Rabi", "Safflower": "Rabi",
    "Sunflower": "Rabi", "Minor Pulses": "Rabi",
    "Sugarcane": "Annual",
}

# Illustrative base prices (Rs/quintal, ~2000 level) used ONLY by the synthetic
# price generator. They are assumptions, not market data.
BASE_PRICE = {
    "Rice": 1000, "Wheat": 1000, "Barley": 800, "Maize": 850, "Sorghum": 900,
    "Kharif Sorghum": 900, "Rabi Sorghum": 950, "Pearl Millet": 850,
    "Finger Millet": 1000, "Chickpea": 2200, "Pigeonpea": 2400,
    "Minor Pulses": 2100, "Groundnut": 2500, "Rapeseed And Mustard": 2300,
    "Sesamum": 3000, "Sunflower": 2400, "Soyabean": 1600, "Safflower": 2200,
    "Linseed": 2100, "Castor": 2300, "Oilseeds": 2200, "Cotton": 2400,
    "Sugarcane": 100,
}
REGION_PRICE_FACTOR = {"North": 0.97, "Central": 1.00, "West": 1.04,
                       "South": 1.05, "East": 0.98, "Northeast": 1.08}

PRICE_YEAR_MIN = 2000
PRICE_ANCHOR_YEAR = 2000
ANNUAL_TREND = 0.02          # paper: 2% annual trend
TEMPORAL_TEST_YEARS = (2016, 2017)   # hold out the last two years (paper: last 2 of 6)
