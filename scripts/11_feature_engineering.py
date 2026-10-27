#Stage 11 — Feature Engineering

#Purpose: Create fraud-analysis features from transaction, 
# temporal, account, and network information while excluding the target and 
# restricted balance variables from the final predictive feature set.

# scripts/11_feature_engineering.py

import pandas as pd
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/interim/cleaned_transactions.csv")
OUTPUT_FILE = Path("data/processed/engineered_transactions.csv")
TABLE_DIR = Path("outputs/tables")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# TEMPORAL FEATURES
# ==============================

df["hour"] = (df["step"] - 1) % 24

df["day"] = ((df["step"] - 1) // 24) + 1

df["day_of_week"] = (df["day"] - 1) % 7

df["is_weekend"] = (
    df["day_of_week"].isin([5, 6])
).astype(int)

# ==============================
# AMOUNT FEATURES
# ==============================

df["amount_log"] = (
    df["amount"] + 1
).apply(lambda x: __import__("math").log1p(x))

df["amount_zero"] = (
    df["amount"] == 0
).astype(int)

# ==============================
# TRANSACTION TYPE FEATURES
# ==============================

df["is_cash_out"] = (
    df["type"] == "CASH_OUT"
).astype(int)

df["is_transfer"] = (
    df["type"] == "TRANSFER"
).astype(int)

df["is_payment"] = (
    df["type"] == "PAYMENT"
).astype(int)

df["is_cash_in"] = (
    df["type"] == "CASH_IN"
).astype(int)

df["is_debit"] = (
    df["type"] == "DEBIT"
).astype(int)

# ==============================
# ACCOUNT FREQUENCY FEATURES
# ==============================

origin_frequency = (
    df["nameOrig"]
    .value_counts()
    .rename("origin_transaction_count")
)

destination_frequency = (
    df["nameDest"]
    .value_counts()
    .rename("destination_transaction_count")
)

df = df.join(
    origin_frequency,
    on="nameOrig"
)

df = df.join(
    destination_frequency,
    on="nameDest"
)

# ==============================
# ACCOUNT DIVERSIFICATION
# ==============================

origin_recipients = (
    df.groupby("nameOrig")["nameDest"]
    .nunique()
    .rename("origin_unique_recipients")
)

destination_senders = (
    df.groupby("nameDest")["nameOrig"]
    .nunique()
    .rename("destination_unique_senders")
)

df = df.join(
    origin_recipients,
    on="nameOrig"
)

df = df.join(
    destination_senders,
    on="nameDest"
)

# ==============================
# TRANSACTION RELATIONSHIP
# ==============================

relationship_frequency = (
    df.groupby(["nameOrig", "nameDest"])
    .size()
    .rename("relationship_transaction_count")
)

df = df.join(
    relationship_frequency,
    on=["nameOrig", "nameDest"]
)

# ==============================
# FLAG INDICATOR
# ==============================

df["flagged_transaction"] = (
    df["isFlaggedFraud"] == 1
).astype(int)

# ==============================
# FINAL FEATURE LIST
# ==============================

# Restricted balance variables are intentionally excluded.
# isFraud is the target and is also excluded from predictors.

excluded_features = [
    "isFraud",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

predictor_features = [
    column
    for column in df.columns
    if column not in excluded_features
]

# ==============================
# FEATURE AUDIT TABLE
# ==============================

feature_audit = pd.DataFrame({
    "feature": df.columns,
    "included_in_predictive_features": [
        column in predictor_features
        for column in df.columns
    ]
})

feature_audit["status"] = feature_audit[
    "feature"
].apply(
    lambda x:
        "TARGET"
        if x == "isFraud"
        else "EXCLUDED_BALANCE"
        if x in excluded_features
        else "CANDIDATE"
)

feature_audit.to_csv(
    TABLE_DIR / "stage_11_feature_list.csv",
    index=False
)

# ==============================
# SAVE ENGINEERED DATA
# ==============================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==============================
# SAVE PREDICTOR LIST
# ==============================

pd.DataFrame({
    "predictor_feature": predictor_features
}).to_csv(
    TABLE_DIR / "stage_11_predictor_features.csv",
    index=False
)

# ==============================
# COMPLETE
# ==============================

print("STAGE 11 COMPLETE")
print(f"Total features: {len(df.columns)}")
print(f"Candidate predictors: {len(predictor_features)}")
print(f"Target: isFraud")
print("Restricted balance variables excluded.")
print(f"Engineered dataset: {OUTPUT_FILE}")