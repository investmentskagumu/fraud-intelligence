# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 20 — COST & DECISION ANALYSIS
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Calculate expected operational cost at each probability
# threshold using:
#
# Expected Cost = (FP × CostFP) + (FN × CostFN)
#
# These costs are scenario assumptions and can be changed
# later using real operational/business estimates.
# ------------------------------------------------------------

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

METRICS_DIR = "outputs/metrics"

threshold_file = os.path.join(
    METRICS_DIR,
    "stage_19_threshold_analysis.csv"
)

output_file = os.path.join(
    METRICS_DIR,
    "stage_20_cost_analysis.csv"
)

summary_file = os.path.join(
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
# LOAD THRESHOLD RESULTS
# ------------------------------------------------------------

df = pd.read_csv(threshold_file)

# ------------------------------------------------------------
# SCENARIO COST ASSUMPTIONS
# ------------------------------------------------------------
# Cost of incorrectly flagging a legitimate transaction.
COST_FP = 1

# Cost of failing to detect a fraudulent transaction.
COST_FN = 10

# ------------------------------------------------------------
# CALCULATE COSTS
# ------------------------------------------------------------

df["Cost_FP"] = (
    df["FP"] * COST_FP
)

df["Cost_FN"] = (
    df["FN"] * COST_FN
)

df["Expected_Cost"] = (
    df["Cost_FP"] +
    df["Cost_FN"]
)

# ------------------------------------------------------------
# NORMALIZED COST
# ------------------------------------------------------------

total_transactions = (
    df["TP"] +
    df["FP"] +
    df["TN"] +
    df["FN"]
)

df["Cost_Per_Transaction"] = (
    df["Expected_Cost"] /
    total_transactions
)

# ------------------------------------------------------------
# COST RANK
# ------------------------------------------------------------

df["Cost_Rank"] = (
    df["Expected_Cost"]
    .rank(
        ascending=True,
        method="min"
    )
)

df = df.sort_values(
    by="Expected_Cost",
    ascending=True
)

# ------------------------------------------------------------
# SAVE COMPLETE ANALYSIS
# ------------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# BEST COST THRESHOLD
# ------------------------------------------------------------

best = df.iloc[0]

# ------------------------------------------------------------
# COST SENSITIVITY
# ------------------------------------------------------------

sensitivity_results = []

for fn_cost in [5, 10, 20, 50, 100]:

    temporary_cost = (
        df["FP"] * COST_FP
        +
        df["FN"] * fn_cost
    )

    best_index = temporary_cost.idxmin()

    row = df.loc[best_index]

    sensitivity_results.append({
        "Cost_FP": COST_FP,
        "Cost_FN": fn_cost,
        "Optimal_Threshold": row["Threshold"],
        "Expected_Cost": temporary_cost.loc[best_index],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1": row["F1"],
        "TP": row["TP"],
        "FP": row["FP"],
        "TN": row["TN"],
        "FN": row["FN"]
    })

sensitivity = pd.DataFrame(
    sensitivity_results
)

# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame([{
    "Cost_FP": COST_FP,
    "Cost_FN": COST_FN,
    "Optimal_Threshold": best["Threshold"],
    "Expected_Cost": best["Expected_Cost"],
    "Cost_Per_Transaction": best["Cost_Per_Transaction"],
    "Precision": best["Precision"],
    "Recall": best["Recall"],
    "F1": best["F1"],
    "Specificity": best["Specificity"],
    "FPR": best["FPR"],
    "TP": best["TP"],
    "FP": best["FP"],
    "TN": best["TN"],
    "FN": best["FN"]
}])

summary = pd.concat(
    [
        summary,
        pd.DataFrame([
            {
                "Cost_FP": row["Cost_FP"],
                "Cost_FN": row["Cost_FN"],
                "Optimal_Threshold": row["Optimal_Threshold"],
                "Expected_Cost": row["Expected_Cost"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1": row["F1"],
                "TP": row["TP"],
                "FP": row["FP"],
                "TN": row["TN"],
                "FN": row["FN"]
            }
            for _, row in sensitivity.iterrows()
        ])
    ],
    ignore_index=True
)

summary.to_csv(
    summary_file,
    index=False
)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 20 — COST & DECISION ANALYSIS")
print("==============================================")

print("\nScenario:")
print(f"False Positive Cost : {COST_FP}")
print(f"False Negative Cost : {COST_FN}")

print("\nOptimal Threshold:")
print(best["Threshold"])

print("\nExpected Cost:")
print(best["Expected_Cost"])

print("\nPerformance at Optimal Threshold:")
print(f"Precision   : {best['Precision']:.4f}")
print(f"Recall      : {best['Recall']:.4f}")
print(f"F1          : {best['F1']:.4f}")
print(f"Specificity : {best['Specificity']:.4f}")
print(f"FPR         : {best['FPR']:.4f}")

print("\nConfusion Matrix:")
print(f"TP : {int(best['TP'])}")
print(f"FP : {int(best['FP'])}")
print(f"TN : {int(best['TN'])}")
print(f"FN : {int(best['FN'])}")

print("\nCost Sensitivity:")
print(
    sensitivity.to_string(index=False)
)

print("\nOutput files:")
print(output_file)
print(summary_file)

print("\nStage 20 completed successfully.")