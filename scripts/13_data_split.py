#Stage 13 — Data Splitting

#Purpose: Create a time-aware train/test split while 
# preserving chronological order and preventing future transactions
# from entering model training

# scripts/13_data_split.py

import pandas as pd
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path(
    "data/processed/engineered_transactions.csv"
)

OUTPUT_DIR = Path("data/processed/splits")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# SORT CHRONOLOGICALLY
# ==============================

df = df.sort_values(
    ["step"]
).reset_index(drop=True)

# ==============================
# TIME-BASED SPLIT
# ==============================

unique_steps = sorted(
    df["step"].unique()
)

split_position = int(
    len(unique_steps) * 0.80
)

train_steps = unique_steps[:split_position]
test_steps = unique_steps[split_position:]

train_df = df[
    df["step"].isin(train_steps)
].copy()

test_df = df[
    df["step"].isin(test_steps)
].copy()

# ==============================
# SAVE SPLITS
# ==============================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

# ==============================
# SPLIT SUMMARY
# ==============================

summary = pd.DataFrame({
    "dataset": [
        "Train",
        "Test"
    ],
    "rows": [
        len(train_df),
        len(test_df)
    ],
    "fraud_transactions": [
        train_df["isFraud"].sum(),
        test_df["isFraud"].sum()
    ],
    "fraud_rate_percentage": [
        train_df["isFraud"].mean() * 100,
        test_df["isFraud"].mean() * 100
    ],
    "minimum_step": [
        train_df["step"].min(),
        test_df["step"].min()
    ],
    "maximum_step": [
        train_df["step"].max(),
        test_df["step"].max()
    ]
})

summary.to_csv(
    "outputs/tables/stage_13_split_summary.csv",
    index=False
)

# ==============================
# VALIDATION
# ==============================

assert train_df["step"].max() < test_df["step"].min()

print("STAGE 13 COMPLETE")
print(f"Train rows: {len(train_df):,}")
print(f"Test rows: {len(test_df):,}")
print(f"Train steps: {train_df['step'].min()} - {train_df['step'].max()}")
print(f"Test steps: {test_df['step'].min()} - {test_df['step'].max()}")
print("Chronological separation validated.")