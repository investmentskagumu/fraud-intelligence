# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 22 — RISK SCORING
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Convert model fraud probabilities into:
#
# 1. Risk score
# 2. Risk category
# 3. Investigation indicator
# 4. Risk rank
#
# Risk categories are derived from the decision threshold
# established in Stage 20.
# ------------------------------------------------------------


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PREDICTIONS_DIR = "outputs/predictions"
METRICS_DIR = "outputs/metrics"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

threshold_file = os.path.join(
    METRICS_DIR,
    "stage_20_cost_summary.csv"
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

if not os.path.exists(threshold_file):
    raise FileNotFoundError(
        f"Required file not found: {threshold_file}"
    )


# ------------------------------------------------------------
# LOAD OPTIMAL THRESHOLD
# ------------------------------------------------------------

cost_summary = pd.read_csv(
    threshold_file
)

if "Optimal_Threshold" not in cost_summary.columns:
    raise ValueError(
        "Optimal_Threshold column not found in Stage 20 summary."
    )

optimal_threshold = float(
    cost_summary.iloc[0]["Optimal_Threshold"]
)


# ------------------------------------------------------------
# FIND STAGE 19 PREDICTION FILE
# ------------------------------------------------------------

prediction_file = os.path.join(
    PREDICTIONS_DIR,
    "stage_19_logistic_predictions.csv"
)

if not os.path.exists(prediction_file):
    raise FileNotFoundError(
        f"Required prediction file not found: {prediction_file}"
    )

predictions = pd.read_csv(
    prediction_file
)


# ------------------------------------------------------------
# IDENTIFY PROBABILITY COLUMN
# ------------------------------------------------------------

probability_candidates = [
    "fraud_probability",
    "probability",
    "y_probability",
    "y_prob",
    "predicted_probability"
]

probability_column = None

for column in probability_candidates:
    if column in predictions.columns:
        probability_column = column
        break

if probability_column is None:
    raise ValueError(
        "Fraud probability column could not be identified."
    )


# ------------------------------------------------------------
# FRAUD PROBABILITY
# ------------------------------------------------------------

predictions["Fraud_Probability"] = pd.to_numeric(
    predictions[probability_column],
    errors="coerce"
).fillna(0)


# ------------------------------------------------------------
# VALIDATE PROBABILITIES
# ------------------------------------------------------------

if (
    (predictions["Fraud_Probability"] < 0).any()
    or
    (predictions["Fraud_Probability"] > 1).any()
):
    raise ValueError(
        "Fraud probabilities must be between 0 and 1."
    )


# ------------------------------------------------------------
# RISK SCORE
# ------------------------------------------------------------
# Scale probability from 0–1 to 0–100.

predictions["Risk_Score"] = (
    predictions["Fraud_Probability"] * 100
)


# ------------------------------------------------------------
# RISK CATEGORIES
# ------------------------------------------------------------
# Categories are based on the selected decision threshold.
#
# Low      : < 25% of threshold
# Medium   : < 50% of threshold
# High     : < 100% of threshold
# Critical : >= decision threshold
# ------------------------------------------------------------

threshold_percent = optimal_threshold * 100

predictions["Risk_Category"] = np.select(
    [
        predictions["Risk_Score"] < threshold_percent * 0.25,
        predictions["Risk_Score"] < threshold_percent * 0.50,
        predictions["Risk_Score"] < threshold_percent,
        predictions["Risk_Score"] >= threshold_percent
    ],
    [
        "Low",
        "Medium",
        "High",
        "Critical"
    ],
    default="Low"
)


# ------------------------------------------------------------
# INVESTIGATION INDICATOR
# ------------------------------------------------------------

predictions["Investigation_Required"] = np.where(
    predictions["Fraud_Probability"]
    >= optimal_threshold,
    "YES",
    "NO"
)


# ------------------------------------------------------------
# RISK RANK
# ------------------------------------------------------------

predictions["Risk_Rank"] = (
    predictions["Fraud_Probability"]
    .rank(
        ascending=False,
        method="first"
    )
    .astype(int)
)


# ------------------------------------------------------------
# SORT BY RISK
# ------------------------------------------------------------

predictions = predictions.sort_values(
    by="Fraud_Probability",
    ascending=False
)


# ------------------------------------------------------------
# SAVE RISK-SCORED DATA
# ------------------------------------------------------------

risk_file = os.path.join(
    PREDICTIONS_DIR,
    "stage_22_risk_scored_transactions.csv"
)

predictions.to_csv(
    risk_file,
    index=False
)


# ------------------------------------------------------------
# RISK CATEGORY SUMMARY
# ------------------------------------------------------------

risk_summary = (
    predictions
    .groupby("Risk_Category")
    .agg(
        Transactions=("Risk_Category", "size"),
        Average_Risk_Score=("Risk_Score", "mean"),
        Maximum_Risk_Score=("Risk_Score", "max"),
        Investigation_Required=(
            "Investigation_Required",
            lambda x: (x == "YES").sum()
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# CATEGORY ORDER
# ------------------------------------------------------------

category_order = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

risk_summary["Risk_Category"] = pd.Categorical(
    risk_summary["Risk_Category"],
    categories=category_order,
    ordered=True
)

risk_summary = risk_summary.sort_values(
    "Risk_Category"
)


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary_file = os.path.join(
    METRICS_DIR,
    "stage_22_risk_summary.csv"
)

risk_summary.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# TOP 100 HIGH-RISK TRANSACTIONS
# ------------------------------------------------------------

top_risk = predictions.head(100)

top_risk_file = os.path.join(
    PREDICTIONS_DIR,
    "stage_22_top_100_risk_transactions.csv"
)

top_risk.to_csv(
    top_risk_file,
    index=False
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 22 — RISK SCORING")
print("==============================================")

print(
    f"\nPrediction File: "
    f"{prediction_file}"
)

print(
    f"\nOptimal Decision Threshold: "
    f"{optimal_threshold:.4f}"
)

print(
    f"Threshold Percentage: "
    f"{threshold_percent:.2f}%"
)

print(
    f"\nTotal Transactions Scored: "
    f"{len(predictions):,}"
)

print(
    f"Transactions Requiring Investigation: "
    f"{(predictions['Investigation_Required'] == 'YES').sum():,}"
)

print("\nRisk Summary:")

print(
    risk_summary.to_string(index=False)
)

print("\nTop 10 Highest-Risk Transactions:")

display_columns = [
    column
    for column in [
        "Risk_Rank",
        "Fraud_Probability",
        "Risk_Score",
        "Risk_Category",
        "Investigation_Required"
    ]
    if column in predictions.columns
]

print(
    predictions[
        display_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

print("\nOutput files:")
print(risk_file)
print(summary_file)
print(top_risk_file)

print("\nStage 22 completed successfully.")