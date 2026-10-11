"""Build the synthetic mandi-price dataset (the paper's data-generating process).

Following Section IV-B of the paper, price is a crop-specific base price plus a 2 %
annual trend, reduced when the realised yield exceeds the crop's base yield, plus
noise. We add a small regional factor and a shared crop-year market shock so that
rows from the same year are correlated, as real mandi prices are.

Every price row keeps the Stage-1 weather/soil features of the real yield record it
was generated from, so the file is self-contained like the paper's dataset.

Usage:  python -m backend.ml.build_price_dataset
"""
import numpy as np
import pandas as pd

from backend.config import (ANNUAL_TREND, BASE_PRICE, PRICE_ANCHOR_YEAR, PRICE_DATA,
                            PRICE_YEAR_MIN, REGION_PRICE_FACTOR, SEED)
from backend.ml.features import load_yield_data


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = load_yield_data()
    df = df[df["year"] >= PRICE_YEAR_MIN].copy()

    base_yield = df.groupby("crop")["yield_qha"].transform("mean")
    ratio = (df["yield_qha"] / base_yield).clip(0.5, 2.0)
    supply = 1.0 - 0.30 * (ratio - 1.0)            # higher yield -> lower price

    base = (df["crop"].map(BASE_PRICE) * df["region"].map(REGION_PRICE_FACTOR)
            * (1 + ANNUAL_TREND) ** (df["year"] - PRICE_ANCHOR_YEAR))

    keys = df[["crop", "year"]].drop_duplicates()
    shock = {tuple(k): rng.normal(0, 0.02) for k in keys.itertuples(index=False)}
    crop_year_shock = np.array([shock[(c, y)] for c, y in zip(df["crop"], df["year"])])

    noise = rng.normal(0, 0.03, len(df))
    df["price_rs_per_quintal"] = (base * supply * (1 + crop_year_shock + noise)).round(2)
    return df.reset_index(drop=True)


def main():
    df = build()
    PRICE_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRICE_DATA, index=False, compression="gzip")
    print(f"price dataset: {len(df)} rows, years {df.year.min()}-{df.year.max()} -> {PRICE_DATA}")
    print(df.groupby("crop")["price_rs_per_quintal"].agg(["mean", "std"]).round(0).head(30))


if __name__ == "__main__":
    main()
