#Stage 03 — Data Quality Audit

#Purpose: Identify missing values, duplicates, invalid values, infinite values, and structural data-quality problems before cleaning.

# 03_data_quality.py

import pandas as pd
import numpy as np
from pathlib import Path

# Purpose: Perform the formal data-quality audit.

DATA_PATH = Path("data/raw/Synthetic_Financial_datasets_log.csv")
OUTPUT_DIR = Path("outputs/tables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

quality = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean() * 100,
    "Unique_Count": df.nunique()
}).reset_index(drop=True)

duplicate_count = df.duplicated().sum()

numeric_columns = df.select_dtypes(include=np.number).columns

infinite_counts = pd.Series({
    col: np.isinf(df[col]).sum()
    for col in numeric_columns
})

quality.to_csv(
    OUTPUT_DIR / "stage_03_data_quality.csv",
    index=False
)

infinite_counts.rename("Infinite_Count").to_csv(
    OUTPUT_DIR / "stage_03_infinite_values.csv"
)

print("STAGE 03 — DATA QUALITY AUDIT")
print("=" * 40)
print(f"Total rows: {len(df):,}")
print(f"Duplicate rows: {duplicate_count:,}")
print(f"Missing values: {df.isnull().sum().sum():,}")
print(f"Negative amounts: {(df['amount'] < 0).sum():,}")
print(f"Invalid fraud values: {(~df['isFraud'].isin([0, 1])).sum():,}")
print(f"Invalid transaction types: {df['type'].isna().sum():,}")

print("\nQuality audit completed.")