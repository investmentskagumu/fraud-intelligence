#Stage 09 — Account Behavior Analysis

#Purpose: Analyze origin and destination account behavior, transaction frequency, 
# amounts, and repeated account activity associated with fraud. 
# Balance variables are used for audit only and are not treated as final predictive features.

import pandas as pd
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/interim/cleaned_transactions.csv")
TABLE_DIR = Path("outputs/tables")

TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# ORIGIN ACCOUNT BEHAVIOR
# ==============================

origin_behavior = (
    df.groupby("nameOrig")
    .agg(
        transaction_count=("nameOrig", "size"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean"),
        maximum_amount=("amount", "max"),
        fraud_count=("isFraud", "sum")
    )
    .reset_index()
)

origin_behavior["fraud_rate"] = (
    origin_behavior["fraud_count"]
    / origin_behavior["transaction_count"]
)

origin_behavior.to_csv(
    TABLE_DIR / "stage_09_origin_account_behavior.csv",
    index=False
)

# ==============================
# DESTINATION ACCOUNT BEHAVIOR
# ==============================

destination_behavior = (
    df.groupby("nameDest")
    .agg(
        transaction_count=("nameDest", "size"),
        total_amount_received=("amount", "sum"),
        average_amount_received=("amount", "mean"),
        maximum_amount_received=("amount", "max"),
        fraud_count=("isFraud", "sum")
    )
    .reset_index()
)

destination_behavior["fraud_rate"] = (
    destination_behavior["fraud_count"]
    / destination_behavior["transaction_count"]
)

destination_behavior.to_csv(
    TABLE_DIR / "stage_09_destination_account_behavior.csv",
    index=False
)

# ==============================
# REPEATED ORIGIN ACCOUNTS
# ==============================

repeated_origins = (
    origin_behavior[
        origin_behavior["transaction_count"] > 1
    ]
    .sort_values("transaction_count", ascending=False)
)

repeated_origins.to_csv(
    TABLE_DIR / "stage_09_repeated_origin_accounts.csv",
    index=False
)

# ==============================
# REPEATED DESTINATION ACCOUNTS
# ==============================

repeated_destinations = (
    destination_behavior[
        destination_behavior["transaction_count"] > 1
    ]
    .sort_values("transaction_count", ascending=False)
)

repeated_destinations.to_csv(
    TABLE_DIR / "stage_09_repeated_destination_accounts.csv",
    index=False
)

# ==============================
# ORIGIN ACCOUNTS WITH FRAUD
# ==============================

fraud_origins = (
    origin_behavior[
        origin_behavior["fraud_count"] > 0
    ]
    .sort_values(
        ["fraud_count", "transaction_count"],
        ascending=False
    )
)

fraud_origins.to_csv(
    TABLE_DIR / "stage_09_fraud_origin_accounts.csv",
    index=False
)

# ==============================
# DESTINATION ACCOUNTS RECEIVING FRAUD
# ==============================

fraud_destinations = (
    destination_behavior[
        destination_behavior["fraud_count"] > 0
    ]
    .sort_values(
        ["fraud_count", "transaction_count"],
        ascending=False
    )
)

fraud_destinations.to_csv(
    TABLE_DIR / "stage_09_fraud_destination_accounts.csv",
    index=False
)

# ==============================
# ACCOUNT TRANSACTION FREQUENCY
# ==============================

origin_frequency = (
    df["nameOrig"]
    .value_counts()
    .rename_axis("nameOrig")
    .reset_index(name="transaction_count")
)

destination_frequency = (
    df["nameDest"]
    .value_counts()
    .rename_axis("nameDest")
    .reset_index(name="transaction_count")
)

origin_frequency.to_csv(
    TABLE_DIR / "stage_09_origin_frequency.csv",
    index=False
)

destination_frequency.to_csv(
    TABLE_DIR / "stage_09_destination_frequency.csv",
    index=False
)

# ==============================
# HIGH-VOLUME ORIGIN ACCOUNTS
# ==============================

high_volume_origins = (
    origin_frequency[
        origin_frequency["transaction_count"] >= 5
    ]
    .sort_values("transaction_count", ascending=False)
)

high_volume_origins.to_csv(
    TABLE_DIR / "stage_09_high_volume_origin_accounts.csv",
    index=False
)

# ==============================
# HIGH-VOLUME DESTINATION ACCOUNTS
# ==============================

high_volume_destinations = (
    destination_frequency[
        destination_frequency["transaction_count"] >= 5
    ]
    .sort_values("transaction_count", ascending=False)
)

high_volume_destinations.to_csv(
    TABLE_DIR / "stage_09_high_volume_destination_accounts.csv",
    index=False
)

# ==============================
# SUMMARY
# ==============================

summary = pd.DataFrame({
    "metric": [
        "Unique origin accounts",
        "Unique destination accounts",
        "Origin accounts with fraud",
        "Destination accounts associated with fraud",
        "Repeated origin accounts",
        "Repeated destination accounts"
    ],
    "value": [
        df["nameOrig"].nunique(),
        df["nameDest"].nunique(),
        (origin_behavior["fraud_count"] > 0).sum(),
        (destination_behavior["fraud_count"] > 0).sum(),
        (origin_behavior["transaction_count"] > 1).sum(),
        (destination_behavior["transaction_count"] > 1).sum()
    ]
})

summary.to_csv(
    TABLE_DIR / "stage_09_account_behavior_summary.csv",
    index=False
)

# ==============================
# COMPLETE
# ==============================

print("STAGE 09 COMPLETE")
print(f"Unique origin accounts: {df['nameOrig'].nunique():,}")
print(f"Unique destination accounts: {df['nameDest'].nunique():,}")
print("Account behavior analysis saved.")