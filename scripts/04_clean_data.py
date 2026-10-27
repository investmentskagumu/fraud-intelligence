#Stage 04 — Data Cleaning

#Purpose: Create a validated cleaned dataset while preserving the raw dataset.

# scripts/04_clean_data.py

import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/raw/Synthetic_Financial_datasets_log.csv")
OUTPUT_FILE = Path("data/interim/cleaned_transactions.csv")
REPORT_FILE = Path("outputs/tables/stage_04_cleaning_report.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)

# ==============================
# CLEANING
# ==============================

# Remove exact duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()

# Replace infinite values with NaN
infinite_count = np.isinf(
    df.select_dtypes(include=np.number)
).sum().sum()

df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing values
missing_before = df.isna().sum().sum()
df = df.dropna().copy()

# Ensure amount is non-negative
negative_amount_count = (df["amount"] < 0).sum()
df = df[df["amount"] >= 0].copy()

# Ensure fraud target is valid
invalid_fraud_count = (~df["isFraud"].isin([0, 1])).sum()
df = df[df["isFraud"].isin([0, 1])].copy()

# ==============================
# SAVE CLEAN DATA
# ==============================

df.to_csv(OUTPUT_FILE, index=False)

# ==============================
# CLEANING REPORT
# ==============================

report = pd.DataFrame({
    "metric": [
        "Original rows",
        "Final rows",
        "Rows removed",
        "Duplicate rows removed",
        "Infinite values found",
        "Missing values found",
        "Negative amounts found",
        "Invalid fraud values found"
    ],
    "value": [
        original_rows,
        len(df),
        original_rows - len(df),
        duplicate_count,
        infinite_count,
        missing_before,
        negative_amount_count,
        invalid_fraud_count
    ]
})

report.to_csv(REPORT_FILE, index=False)

print("STAGE 04 COMPLETE")
print(f"Original rows: {original_rows:,}")
print(f"Final rows: {len(df):,}")
print(f"Rows removed: {original_rows - len(df):,}")
print(f"Clean dataset: {OUTPUT_FILE}")