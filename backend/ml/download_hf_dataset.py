"""Download the Hugging Face crop-yield dataset to data/yield_raw.csv.

    pip install -r requirements-hf.txt
    python -m backend.ml.download_hf_dataset
    python -m backend.ml.prepare_yield_data

Note: the repo already ships data/yield_processed.csv.gz, parsed from the same
prompt/completion CSV, so this step is only needed to refresh the data. The parser
expects the columns: prompt, state_name, dist_name, year, crop, yield_kg_per_ha.
"""
from datasets import load_dataset

from backend.config import YIELD_RAW_CSV

if __name__ == "__main__":
    ds = load_dataset("Gamestatue1/indian-crop-yield-prediction")
    split = "train" if "train" in ds else next(iter(ds))
    YIELD_RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    ds[split].to_pandas().to_csv(YIELD_RAW_CSV, index=False)
    print(f"saved {len(ds[split])} rows -> {YIELD_RAW_CSV}")
