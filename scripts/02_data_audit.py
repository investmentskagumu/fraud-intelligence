#%% Stage 02 — Data Structure Audit

# Purpose: Determine the actual structure, variables, data types, ranges, categories, identifiers, target, and time range present in the dataset.

#%%

# 01_load_data.py

import pandas as pd
from pathlib import Path

# Purpose: Load the raw PaySim dataset and create an initial dataset metadata report.

DATA_PATH = Path("data/raw/Synthetic_Financial_datasets_log.csv")
OUTPUT_PATH = Path("outputs/tables/stage_01_metadata.csv")

df = pd.read_csv(DATA_PATH)

metadata = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "File",
        "Memory Usage (MB)"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        str(DATA_PATH),
        round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
    ]
})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
metadata.to_csv(OUTPUT_PATH, index=False)

print("STAGE 01 — DATA INGESTION")
print("=" * 40)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Dataset loaded successfully.")
print(f"Metadata saved to: {OUTPUT_PATH}")