#Stage 05 — Target & Class Analysis

#Purpose: Analyze the fraud target (isFraud), fraud prevalence, 
# and the relationship between actual fraud and isFlaggedFraud.

# scripts/05_target_analysis.py

import pandas as pd
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/interim/cleaned_transactions.csv")
OUTPUT_DIR = Path("outputs/tables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("data/raw/Synthetic_Financial_datasets_log.csv")

# ==============================
# TARGET ANALYSIS
# ==============================

total_transactions = len(df)

fraud_count = (df["isFraud"] == 1).sum()
legitimate_count = (df["isFraud"] == 0).sum()

fraud_percentage = (fraud_count / total_transactions) * 100
legitimate_percentage = (legitimate_count / total_transactions) * 100

# ==============================
# CLASS DISTRIBUTION
# ==============================

class_distribution = (
    df["isFraud"]
    .value_counts()
    .rename_axis("isFraud")
    .reset_index(name="transaction_count")
)

class_distribution["percentage"] = (
    class_distribution["transaction_count"]
    / total_transactions
) * 100

class_distribution.to_csv(
    OUTPUT_DIR / "stage_05_class_distribution.csv",
    index=False
)

# ==============================
# FRAUD BY TRANSACTION TYPE
# ==============================

fraud_by_type = (
    df.groupby("type")
    .agg(
        total_transactions=("isFraud", "count"),
        fraud_transactions=("isFraud", "sum")
    )
    .reset_index()
)

fraud_by_type["fraud_rate_percentage"] = (
    fraud_by_type["fraud_transactions"]
    / fraud_by_type["total_transactions"]
) * 100

fraud_by_type.to_csv(
    OUTPUT_DIR / "stage_05_fraud_by_type.csv",
    index=False
)

# ==============================
# FLAGGED FRAUD ANALYSIS
# ==============================

flagged_total = (df["isFlaggedFraud"] == 1).sum()

flagged_actual_fraud = (
    (df["isFlaggedFraud"] == 1) &
    (df["isFraud"] == 1)
).sum()

unflagged_actual_fraud = (
    (df["isFlaggedFraud"] == 0) &
    (df["isFraud"] == 1)
).sum()

flag_analysis = pd.DataFrame({
    "metric": [
        "Total transactions",
        "Legitimate transactions",
        "Fraudulent transactions",
        "Legitimate percentage",
        "Fraud percentage",
        "Flagged transactions",
        "Flagged transactions that were actually fraud",
        "Fraud transactions not flagged"
    ],
    "value": [
        total_transactions,
        legitimate_count,
        fraud_count,
        legitimate_percentage,
        fraud_percentage,
        flagged_total,
        flagged_actual_fraud,
        unflagged_actual_fraud
    ]
})

flag_analysis.to_csv(
    OUTPUT_DIR / "stage_05_flag_analysis.csv",
    index=False
)

# ==============================
# PRINT RESULTS
# ==============================

print("STAGE 05 COMPLETE")
print(f"Total transactions: {total_transactions:,}")
print(f"Legitimate transactions: {legitimate_count:,}")
print(f"Fraudulent transactions: {fraud_count:,}")
print(f"Fraud rate: {fraud_percentage:.4f}%")
print(f"Flagged transactions: {flagged_total:,}")
print(f"Actual fraud not flagged: {unflagged_actual_fraud:,}")