#Stage 10 — Recipient / Network Analysis

#Purpose: Analyze sender–recipient relationships, repeated transfers, 
# recipient concentration, and transaction-network patterns associated with fraud

# scripts/10_network_analysis.py

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
# SENDER → RECIPIENT RELATIONSHIPS
# ==============================

relationships = (
    df.groupby(["nameOrig", "nameDest"])
    .agg(
        transaction_count=("amount", "size"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean"),
        fraud_count=("isFraud", "sum")
    )
    .reset_index()
)

relationships["fraud_rate"] = (
    relationships["fraud_count"]
    / relationships["transaction_count"]
)

relationships.to_csv(
    TABLE_DIR / "stage_10_sender_recipient_relationships.csv",
    index=False
)

# ==============================
# REPEATED RELATIONSHIPS
# ==============================

repeated_relationships = (
    relationships[
        relationships["transaction_count"] > 1
    ]
    .sort_values(
        "transaction_count",
        ascending=False
    )
)

repeated_relationships.to_csv(
    TABLE_DIR / "stage_10_repeated_relationships.csv",
    index=False
)

# ==============================
# FRAUDULENT RELATIONSHIPS
# ==============================

fraud_relationships = (
    relationships[
        relationships["fraud_count"] > 0
    ]
    .sort_values(
        ["fraud_count", "transaction_count"],
        ascending=False
    )
)

fraud_relationships.to_csv(
    TABLE_DIR / "stage_10_fraud_relationships.csv",
    index=False
)

# ==============================
# RECIPIENT CONCENTRATION
# ==============================

recipient_concentration = (
    df.groupby("nameDest")
    .agg(
        unique_senders=("nameOrig", "nunique"),
        transaction_count=("nameDest", "size"),
        total_received=("amount", "sum"),
        fraud_count=("isFraud", "sum")
    )
    .reset_index()
)

recipient_concentration["fraud_rate"] = (
    recipient_concentration["fraud_count"]
    / recipient_concentration["transaction_count"]
)

recipient_concentration = recipient_concentration.sort_values(
    "unique_senders",
    ascending=False
)

recipient_concentration.to_csv(
    TABLE_DIR / "stage_10_recipient_concentration.csv",
    index=False
)

# ==============================
# MULTI-SENDER RECIPIENTS
# ==============================

multi_sender_recipients = (
    recipient_concentration[
        recipient_concentration["unique_senders"] > 1
    ]
)

multi_sender_recipients.to_csv(
    TABLE_DIR / "stage_10_multi_sender_recipients.csv",
    index=False
)

# ==============================
# SENDER DIVERSIFICATION
# ==============================

sender_diversification = (
    df.groupby("nameOrig")
    .agg(
        unique_recipients=("nameDest", "nunique"),
        transaction_count=("nameOrig", "size"),
        total_sent=("amount", "sum"),
        fraud_count=("isFraud", "sum")
    )
    .reset_index()
)

sender_diversification["fraud_rate"] = (
    sender_diversification["fraud_count"]
    / sender_diversification["transaction_count"]
)

sender_diversification = sender_diversification.sort_values(
    "unique_recipients",
    ascending=False
)

sender_diversification.to_csv(
    TABLE_DIR / "stage_10_sender_diversification.csv",
    index=False
)

# ==============================
# MULTI-RECIPIENT SENDERS
# ==============================

multi_recipient_senders = (
    sender_diversification[
        sender_diversification["unique_recipients"] > 1
    ]
)

multi_recipient_senders.to_csv(
    TABLE_DIR / "stage_10_multi_recipient_senders.csv",
    index=False
)

# ==============================
# FRAUD NETWORK SUMMARY
# ==============================

network_summary = pd.DataFrame({
    "metric": [
        "Unique sender-recipient relationships",
        "Repeated relationships",
        "Fraudulent relationships",
        "Recipients with multiple senders",
        "Senders with multiple recipients",
        "Unique recipients",
        "Unique senders"
    ],
    "value": [
        len(relationships),
        len(repeated_relationships),
        len(fraud_relationships),
        len(multi_sender_recipients),
        len(multi_recipient_senders),
        df["nameDest"].nunique(),
        df["nameOrig"].nunique()
    ]
})

network_summary.to_csv(
    TABLE_DIR / "stage_10_network_summary.csv",
    index=False
)

# ==============================
# COMPLETE
# ==============================

print("STAGE 10 COMPLETE")
print(f"Unique relationships: {len(relationships):,}")
print(f"Fraudulent relationships: {len(fraud_relationships):,}")
print("Network analysis tables saved.")