# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 23 — INVESTIGATION PRIORITIZATION
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Prioritize transactions for fraud investigation using:
#
# 1. Fraud probability
# 2. Risk score
# 3. Risk category
# 4. Investigation indicator
#
# The highest-risk transactions are ranked first.
# ------------------------------------------------------------


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PREDICTIONS_DIR = "outputs/predictions"
METRICS_DIR = "outputs/metrics"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

input_file = os.path.join(
    PREDICTIONS_DIR,
    "stage_22_risk_scored_transactions.csv"
)

output_file = os.path.join(
    PREDICTIONS_DIR,
    "stage_23_investigation_priority.csv"
)

summary_file = os.path.join(
    METRICS_DIR,
    "stage_23_investigation_summary.csv"
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Required file not found: {input_file}"
    )


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(input_file)


# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "Fraud_Probability",
    "Risk_Score",
    "Risk_Category",
    "Investigation_Required",
    "Risk_Rank"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# VALIDATE DATA
# ------------------------------------------------------------

df["Fraud_Probability"] = pd.to_numeric(
    df["Fraud_Probability"],
    errors="coerce"
).fillna(0)

df["Risk_Score"] = pd.to_numeric(
    df["Risk_Score"],
    errors="coerce"
).fillna(0)


# ------------------------------------------------------------
# INVESTIGATION PRIORITY
# ------------------------------------------------------------
# Priority is based primarily on fraud probability.
#
# Critical = highest priority
# High     = high priority
# Medium   = moderate priority
# Low      = lowest priority
# ------------------------------------------------------------

priority_map = {
    "Critical": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4
}

df["Priority_Level"] = (
    df["Risk_Category"]
    .map(priority_map)
    .fillna(5)
    .astype(int)
)


# ------------------------------------------------------------
# INVESTIGATION STATUS
# ------------------------------------------------------------

df["Investigation_Priority"] = np.select(
    [
        df["Risk_Category"] == "Critical",
        df["Risk_Category"] == "High",
        df["Risk_Category"] == "Medium",
        df["Risk_Category"] == "Low"
    ],
    [
        "URGENT",
        "HIGH",
        "MODERATE",
        "LOW"
    ],
    default="LOW"
)


# ------------------------------------------------------------
# INVESTIGATION RANK
# ------------------------------------------------------------

df = df.sort_values(
    by=[
        "Fraud_Probability",
        "Risk_Score"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)

df["Investigation_Rank"] = (
    df.index + 1
)


# ------------------------------------------------------------
# INVESTIGATION QUEUE
# ------------------------------------------------------------
# Transactions marked YES in Stage 22 are placed into
# the active investigation queue.

df["Investigation_Queue"] = np.where(
    df["Investigation_Required"] == "YES",
    "ACTIVE",
    "MONITOR"
)


# ------------------------------------------------------------
# QUEUE RANK
# ------------------------------------------------------------
# Rank only transactions requiring investigation.

investigation_mask = (
    df["Investigation_Required"] == "YES"
)

df["Investigation_Queue_Rank"] = np.nan

df.loc[
    investigation_mask,
    "Investigation_Queue_Rank"
] = (
    df.loc[
        investigation_mask,
        "Fraud_Probability"
    ]
    .rank(
        ascending=False,
        method="first"
    )
    .astype(int)
)


# ------------------------------------------------------------
# SAVE PRIORITIZED TRANSACTIONS
# ------------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# INVESTIGATION SUMMARY
# ------------------------------------------------------------

summary = (
    df.groupby(
        [
            "Risk_Category",
            "Investigation_Priority",
            "Investigation_Queue"
        ]
    )
    .agg(
        Transactions=("Risk_Category", "size"),
        Average_Risk_Score=("Risk_Score", "mean"),
        Maximum_Risk_Score=("Risk_Score", "max"),
        Average_Fraud_Probability=(
            "Fraud_Probability",
            "mean"
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# CATEGORY ORDER
# ------------------------------------------------------------

category_order = [
    "Critical",
    "High",
    "Medium",
    "Low"
]

summary["Risk_Category"] = pd.Categorical(
    summary["Risk_Category"],
    categories=category_order,
    ordered=True
)

summary = summary.sort_values(
    "Risk_Category"
)


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 23 — INVESTIGATION PRIORITIZATION")
print("==============================================")

print(
    f"\nTotal Transactions: "
    f"{len(df):,}"
)

print(
    f"Active Investigation Queue: "
    f"{investigation_mask.sum():,}"
)

print(
    f"Transactions on Monitor: "
    f"{(~investigation_mask).sum():,}"
)

print("\nInvestigation Summary:")

print(
    summary.to_string(index=False)
)


# ------------------------------------------------------------
# TOP INVESTIGATION CASES
# ------------------------------------------------------------

top_cases = df[
    df["Investigation_Required"] == "YES"
].head(20)

display_columns = [
    column
    for column in [
        "Investigation_Queue_Rank",
        "Risk_Rank",
        "Fraud_Probability",
        "Risk_Score",
        "Risk_Category",
        "Investigation_Priority",
        "Investigation_Required"
    ]
    if column in top_cases.columns
]

print("\nTop 20 Investigation Cases:")

print(
    top_cases[
        display_columns
    ].to_string(index=False)
)


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

print("\nOutput files:")
print(output_file)
print(summary_file)

print("\nStage 23 completed successfully.")