"""Parse the crop-yield CSV (prompt/completion format) into a tidy numeric table.

Soil and weather values only exist inside the natural-language `prompt` column,
so they are extracted with regular expressions.

Usage:
    python -m backend.ml.prepare_yield_data [path/to/raw.csv|raw.csv.gz]
"""
import re
import sys
from pathlib import Path

import pandas as pd

from backend.config import REGION, SEASON, YIELD_DATA, YIELD_RAW_CSV

# The CSV mixes several prompt templates, so each field has alternative patterns.
RX = {
    "soil_ph": [r"pH of ([\d.]+)", r"pH ([\d.]+) \("],
    "organic_carbon_gkg": [r"([\d.]+) g/kg (?:soil )?organic carbon",
                           r"organic carbon ([\d.]+) g/kg"],
    "rainfall_mm": [r"about (\d+(?:\.\d+)?)\s*mm of total rainfall",
                    r"(\d+(?:\.\d+)?)\s*mm total rainfall"],
    "temp_avg_c": [r"average temperature of (-?[\d.]+)\s*°C", r"avg temp (-?[\d.]+)\s*°C"],
    "area_kha": [r"About ([\d.]+) thousand hectares", r"Area under cultivation: ([\d.]+) thousand"],
}
RX_RANGE = [r"ranging from (-?[\d.]+)\s*°C to (-?[\d.]+)\s*°C",
            r"range (-?[\d.]+)-(-?[\d.]+)\s*°C"]
RX_TEXTURE = [r"is ([a-z\-]+) in texture", r"[,(] ?([a-z\-]+) texture"]


def _first(series: pd.Series, patterns: list[str]) -> pd.Series:
    out = pd.Series(index=series.index, dtype=object)
    for rx in patterns:
        ext = series.str.extract(rx)
        out = out.fillna(ext[0]) if ext.shape[1] == 1 else out
    return out


def parse(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw[["state_name", "dist_name", "year", "crop", "yield_kg_per_ha"]].copy()
    prompt = raw["prompt"]
    for col, pats in RX.items():
        out[col] = _first(prompt, pats)
    tmin = pd.Series(index=prompt.index, dtype=object)
    tmax = pd.Series(index=prompt.index, dtype=object)
    for rx in RX_RANGE:
        ext = prompt.str.extract(rx)
        tmin, tmax = tmin.fillna(ext[0]), tmax.fillna(ext[1])
    out["temp_min_c"], out["temp_max_c"] = tmin, tmax
    out["soil_texture"] = _first(prompt, RX_TEXTURE)
    for col in ["year", "soil_ph", "organic_carbon_gkg", "rainfall_mm", "temp_avg_c",
                "temp_min_c", "temp_max_c", "area_kha", "yield_kg_per_ha"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.rename(columns={"state_name": "state", "dist_name": "district"})
    out["yield_qha"] = out["yield_kg_per_ha"] / 100.0      # kg/ha -> quintal/ha
    out["temp_range_c"] = out["temp_max_c"] - out["temp_min_c"]
    out["region"] = out["state"].map(REGION)
    out["season"] = out["crop"].map(SEASON)
    return out.drop(columns=["yield_kg_per_ha"])


def main(path: str | None = None) -> None:
    src = Path(path) if path else YIELD_RAW_CSV
    if not src.exists():
        sys.exit(f"Raw yield file not found: {src}\n"
                 "Pass its path as an argument or place it at data/yield_raw.csv")
    raw = pd.read_csv(src)
    df = parse(raw)
    n0 = len(df)
    df = df.dropna(subset=["region", "season", "yield_qha"])
    df = df[df["yield_qha"] > 0]
    YIELD_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(YIELD_DATA, index=False, compression="gzip")
    w = df["rainfall_mm"].notna().sum()
    print(f"parsed {n0} rows -> kept {len(df)} ({w} with weather) -> {YIELD_DATA}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
