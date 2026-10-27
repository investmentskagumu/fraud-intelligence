#Stage 06 — Exploratory Data Analysis (EDA)

#Purpose: Explore transaction distributions, amounts, transaction types, and 
# fraud-related patterns using descriptive analysis and visualizations.

# scripts/06_eda.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/interim/cleaned_transactions.csv")
FIGURE_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("data/raw/Synthetic_Financial_datasets_log.csv")

# ==============================
# BASIC DESCRIPTIVE STATISTICS
# ==============================

numeric_summary = df[
    ["step", "amount",
     "oldbalanceOrg", "newbalanceOrig",
     "oldbalanceDest", "newbalanceDest"]
].describe()

numeric_summary.to_csv(
    TABLE_DIR / "stage_06_numeric_summary.csv"
)

# ==============================
# TRANSACTION TYPE DISTRIBUTION
# ==============================

type_distribution = (
    df["type"]
    .value_counts()
    .reset_index()
)

type_distribution.columns = ["transaction_type", "transaction_count"]

type_distribution.to_csv(
    TABLE_DIR / "stage_06_transaction_type_distribution.csv",
    index=False
)

plt.figure(figsize=(10, 6))
type_distribution.plot(
    x="transaction_type",
    y="transaction_count",
    kind="bar",
    legend=False
)

plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_06_transaction_type_distribution.png"
)

plt.close()

# ==============================
# TRANSACTION AMOUNT DISTRIBUTION
# ==============================

plt.figure(figsize=(10, 6))

plt.hist(
    df["amount"],
    bins=100
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_06_amount_distribution.png"
)

plt.close()

# ==============================
# LOG AMOUNT DISTRIBUTION
# ==============================

plt.figure(figsize=(10, 6))

plt.hist(
    (df["amount"] + 1).apply(lambda x: __import__("math").log10(x)),
    bins=100
)

plt.title("Log-Scaled Transaction Amount Distribution")
plt.xlabel("log10(Amount + 1)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_06_log_amount_distribution.png"
)

plt.close()

# ==============================
# FRAUD VS LEGITIMATE AMOUNTS
# ==============================

fraud_amount_summary = (
    df.groupby("isFraud")["amount"]
    .agg(["count", "mean", "median", "min", "max", "std"])
    .reset_index()
)

fraud_amount_summary.to_csv(
    TABLE_DIR / "stage_06_fraud_amount_summary.csv",
    index=False
)

# ==============================
# FRAUD BY TRANSACTION TYPE
# ==============================

fraud_type = pd.crosstab(
    df["type"],
    df["isFraud"]
)

fraud_type.to_csv(
    TABLE_DIR / "stage_06_fraud_transaction_type.csv"
)

# ==============================
# STEP DISTRIBUTION
# ==============================

step_summary = (
    df.groupby("step")
    .agg(
        transactions=("step", "size"),
        fraud_transactions=("isFraud", "sum")
    )
    .reset_index()
)

step_summary["fraud_rate"] = (
    step_summary["fraud_transactions"]
    / step_summary["transactions"]
)

step_summary.to_csv(
    TABLE_DIR / "stage_06_step_summary.csv",
    index=False
)

# ==============================
# FRAUD RATE BY TRANSACTION TYPE
# ==============================

fraud_rate_type = (
    df.groupby("type")
    .agg(
        transactions=("isFraud", "size"),
        fraud_transactions=("isFraud", "sum")
    )
    .reset_index()
)

fraud_rate_type["fraud_rate_percentage"] = (
    fraud_rate_type["fraud_transactions"]
    / fraud_rate_type["transactions"]
) * 100

fraud_rate_type.to_csv(
    TABLE_DIR / "stage_06_fraud_rate_by_type.csv",
    index=False
)

# ==============================
# COMPLETE
# ==============================

print("STAGE 06 COMPLETE")
print(f"Rows analyzed: {len(df):,}")
print("EDA tables saved.")
print("EDA figures saved.")