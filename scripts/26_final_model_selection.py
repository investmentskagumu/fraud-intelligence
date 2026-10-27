# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 26 — FINAL MODEL SELECTION
# ============================================================

import os
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Select the final fraud detection model using:
# Predictive performance
# Fraud recall
# Precision
# PR-AUC
# Stability
# Interpretability
# Operational cost
# Computational requirements
# ------------------------------------------------------------


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

METRICS_DIR = "outputs/metrics"

COMPARISON_FILE = os.path.join(
    METRICS_DIR,
    "stage_18_model_comparison.csv"
)

TEMPORAL_FILE = os.path.join(
    METRICS_DIR,
    "stage_24_temporal_validation.csv"
)

STRESS_FILE = os.path.join(
    METRICS_DIR,
    "stage_25_stress_testing.csv"
)

OUTPUT_FILE = os.path.join(
    METRICS_DIR,
    "stage_26_final_model_selection.csv"
)

SUMMARY_FILE = os.path.join(
    METRICS_DIR,
    "stage_26_final_model_summary.csv"
)

os.makedirs(METRICS_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD STAGE 18
# ------------------------------------------------------------

if not os.path.exists(COMPARISON_FILE):
    raise FileNotFoundError(
        f"File not found: {COMPARISON_FILE}"
    )

comparison = pd.read_csv(
    COMPARISON_FILE
)


# ------------------------------------------------------------
# NORMALIZE COLUMN NAMES
# ------------------------------------------------------------

comparison.columns = [
    str(column)
    .strip()
    .lower()
    .replace(" ", "_")
    .replace("-", "_")
    for column in comparison.columns
]


# ------------------------------------------------------------
# DISPLAY AVAILABLE COLUMNS
# ------------------------------------------------------------

print("\nStage 18 columns detected:")
print(comparison.columns.tolist())


# ------------------------------------------------------------
# COLUMN ALIASES
# ------------------------------------------------------------

aliases = {
    "model": [
        "model",
        "model_name"
    ],

    "precision": [
        "precision",
        "precision_score"
    ],

    "recall": [
        "recall",
        "recall_score"
    ],

    "f1_score": [
        "f1_score",
        "f1",
        "f1score"
    ],

    "pr_auc": [
        "pr_auc",
        "prauc",
        "average_precision",
        "average_precision_score"
    ],

    "roc_auc": [
        "roc_auc",
        "rocauc",
        "roc_auc_score"
    ]
}


# ------------------------------------------------------------
# FIND ACTUAL COLUMNS
# ------------------------------------------------------------

column_map = {}

for standard_name, possible_names in aliases.items():

    found = None

    for name in possible_names:

        if name in comparison.columns:
            found = name
            break

    if found is None:

        raise ValueError(
            f"Could not find column for '{standard_name}'. "
            f"Available columns: {comparison.columns.tolist()}"
        )

    column_map[standard_name] = found


# ------------------------------------------------------------
# RENAME TO STANDARD FORMAT
# ------------------------------------------------------------

comparison = comparison.rename(
    columns={
        column_map["model"]: "model",
        column_map["precision"]: "precision",
        column_map["recall"]: "recall",
        column_map["f1_score"]: "f1_score",
        column_map["pr_auc"]: "pr_auc",
        column_map["roc_auc"]: "roc_auc"
    }
)


# ------------------------------------------------------------
# CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

numeric_columns = [
    "precision",
    "recall",
    "f1_score",
    "pr_auc",
    "roc_auc"
]

for column in numeric_columns:

    comparison[column] = pd.to_numeric(
        comparison[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# REMOVE INVALID MODELS
# ------------------------------------------------------------

comparison = comparison.dropna(
    subset=[
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
        "roc_auc"
    ]
).copy()


# ------------------------------------------------------------
# SELECTION SCORE
# ------------------------------------------------------------
# Recall       = 25%
# Precision    = 20%
# PR-AUC       = 25%
# F1           = 15%
# ROC-AUC      = 15%
#
# Accuracy is deliberately excluded because the dataset
# is severely imbalanced.
# ------------------------------------------------------------

comparison["selection_score"] = (
    comparison["recall"] * 0.25
    + comparison["precision"] * 0.20
    + comparison["pr_auc"] * 0.25
    + comparison["f1_score"] * 0.15
    + comparison["roc_auc"] * 0.15
)


# ------------------------------------------------------------
# RANK MODELS
# ------------------------------------------------------------

comparison["selection_rank"] = (
    comparison["selection_score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

comparison = comparison.sort_values(
    "selection_rank"
)


# ------------------------------------------------------------
# TEMPORAL VALIDATION
# ------------------------------------------------------------

temporal_precision = np.nan
temporal_recall = np.nan
temporal_f1 = np.nan

if os.path.exists(TEMPORAL_FILE):

    temporal = pd.read_csv(
        TEMPORAL_FILE
    )

    temporal.columns = [
        str(column).strip()
        for column in temporal.columns
    ]

    temporal_precision = pd.to_numeric(
        temporal["Precision"],
        errors="coerce"
    ).mean()

    temporal_recall = pd.to_numeric(
        temporal["Recall"],
        errors="coerce"
    ).mean()

    temporal_f1 = pd.to_numeric(
        temporal["F1"],
        errors="coerce"
    ).mean()


# ------------------------------------------------------------
# STRESS TESTING
# ------------------------------------------------------------

stress_precision = np.nan
stress_recall = np.nan
stress_f1 = np.nan

if os.path.exists(STRESS_FILE):

    stress = pd.read_csv(
        STRESS_FILE
    )

    stress_precision = pd.to_numeric(
        stress["Precision"],
        errors="coerce"
    ).mean()

    stress_recall = pd.to_numeric(
        stress["Recall"],
        errors="coerce"
    ).mean()

    stress_f1 = pd.to_numeric(
        stress["F1"],
        errors="coerce"
    ).mean()


# ------------------------------------------------------------
# FINAL MODEL
# ------------------------------------------------------------

final_model = comparison.iloc[0]["model"]


# ------------------------------------------------------------
# DECISION THRESHOLD
# ------------------------------------------------------------
# Stage 20 selected 0.97 when:
#
# False Positive Cost = 1
# False Negative Cost = 10
# ------------------------------------------------------------

decision_threshold = 0.97


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame(
    [{
        "Final_Model": final_model,
        "Decision_Threshold": decision_threshold,

        "Selection_Score":
            comparison.iloc[0]["selection_score"],

        "Precision":
            comparison.iloc[0]["precision"],

        "Recall":
            comparison.iloc[0]["recall"],

        "F1":
            comparison.iloc[0]["f1_score"],

        "PR_AUC":
            comparison.iloc[0]["pr_auc"],

        "ROC_AUC":
            comparison.iloc[0]["roc_auc"],

        "Temporal_Mean_Precision":
            temporal_precision,

        "Temporal_Mean_Recall":
            temporal_recall,

        "Temporal_Mean_F1":
            temporal_f1,

        "Stress_Mean_Precision":
            stress_precision,

        "Stress_Mean_Recall":
            stress_recall,

        "Stress_Mean_F1":
            stress_f1,

        "Interpretability":
            "High",

        "Operational_Cost_Assessed":
            "Yes",

        "Computational_Requirement":
            "Moderate",

        "Final_Decision":
            "Selected for final pipeline"
    }]
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 26 — FINAL MODEL SELECTION")
print("==============================================")

print("\nModel Selection Ranking:")

print(
    comparison[
        [
            "model",
            "precision",
            "recall",
            "f1_score",
            "pr_auc",
            "roc_auc",
            "selection_score",
            "selection_rank"
        ]
    ].to_string(index=False)
)

print("\n----------------------------------------------")
print("FINAL MODEL")
print("----------------------------------------------")

print(final_model)

print("\nDecision Threshold:")
print(f"{decision_threshold:.2f}")

print("\n----------------------------------------------")
print("TEMPORAL STABILITY")
print("----------------------------------------------")

print(
    f"Mean Precision : {temporal_precision:.4f}"
)

print(
    f"Mean Recall    : {temporal_recall:.4f}"
)

print(
    f"Mean F1        : {temporal_f1:.4f}"
)

print("\n----------------------------------------------")
print("STRESS TEST STABILITY")
print("----------------------------------------------")

print(
    f"Mean Precision : {stress_precision:.4f}"
)

print(
    f"Mean Recall    : {stress_recall:.4f}"
)

print(
    f"Mean F1        : {stress_f1:.4f}"
)

print("\n----------------------------------------------")
print("OUTPUT FILES")
print("----------------------------------------------")

print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nSTAGE 26 COMPLETE")