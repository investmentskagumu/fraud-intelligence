#Stage 07 — Fraud Pattern Analysis

#Purpose: Identify transaction types, amounts, timing, and 
# behavioral patterns that distinguish fraudulent transactions from legitimate transactions.

#scripts/07_fraud_patterns.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/interim/cleaned_transactions.csv")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# FRAUD BY TRANSACTION TYPE
# ==============================

fraud_type = (
    df.groupby("type")
    .agg(
        total_transactions=("isFraud", "count"),
        fraudulent_transactions=("isFraud", "sum")
    )
    .reset_index()
)

fraud_type["fraud_rate"] = (
    fraud_type["fraudulent_transactions"]
    / fraud_type["total_transactions"]
) * 100

fraud_type.to_csv(
    TABLE_DIR / "stage_07_fraud_by_type.csv",
    index=False
)

# ==============================
# FRAUD AMOUNT ANALYSIS
# ==============================

fraud_amount = (
    df.groupby("isFraud")["amount"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max",
        standard_deviation="std"
    )
    .reset_index()
)

fraud_amount.to_csv(
    TABLE_DIR / "stage_07_fraud_amount_analysis.csv",
    index=False
)

# ==============================
# FRAUD AMOUNT QUANTILES
# ==============================

fraud_transactions = df[df["isFraud"] == 1].copy()

amount_quantiles = (
    fraud_transactions["amount"]
    .quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    .reset_index()
)

amount_quantiles.columns = ["quantile", "amount"]

amount_quantiles.to_csv(
    TABLE_DIR / "stage_07_fraud_amount_quantiles.csv",
    index=False
)

# ==============================
# FRAUD BY STEP
# ==============================

fraud_step = (
    df.groupby("step")
    .agg(
        total_transactions=("isFraud", "count"),
        fraudulent_transactions=("isFraud", "sum")
    )
    .reset_index()
)

fraud_step["fraud_rate"] = (
    fraud_step["fraudulent_transactions"]
    / fraud_step["total_transactions"]
) * 100

fraud_step.to_csv(
    TABLE_DIR / "stage_07_fraud_by_step.csv",
    index=False
)

# ==============================
# FRAUD BY TRANSACTION TYPE
# VISUALIZATION
# ==============================

plt.figure(figsize=(10, 6))

plt.bar(
    fraud_type["type"],
    fraud_type["fraud_rate"]
)

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_07_fraud_rate_by_type.png"
)

plt.close()

# ==============================
# FRAUD BY STEP
# VISUALIZATION
# ==============================

plt.figure(figsize=(12, 6))

plt.plot(
    fraud_step["step"],
    fraud_step["fraud_rate"]
)

plt.title("Fraud Rate Across Transaction Steps")
plt.xlabel("Step")
plt.ylabel("Fraud Rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_07_fraud_rate_by_step.png"
)

plt.close()

# ==============================
# FRAUD AMOUNT DISTRIBUTION
# ==============================

plt.figure(figsize=(10, 6))

plt.hist(
    fraud_transactions["amount"],
    bins=50
)

plt.title("Fraudulent Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_07_fraud_amount_distribution.png"
)

plt.close()

# ==============================
# TOP FRAUD TRANSACTIONS
# ==============================

top_fraud = (
    fraud_transactions
    .sort_values("amount", ascending=False)
    .head(100)
)

top_fraud.to_csv(
    TABLE_DIR / "stage_07_top_100_fraud_transactions.csv",
    index=False
)

# ==============================
# COMPLETE
# ==============================

print("STAGE 07 COMPLETE")
print(f"Fraudulent transactions analyzed: {len(fraud_transactions):,}")
print("Fraud pattern tables saved.")
print("Fraud pattern figures saved.")