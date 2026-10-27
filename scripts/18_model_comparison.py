# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 18 — MODEL COMPARISON
# ============================================================

import os
import pandas as pd

# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Compare model performance from Stages 15, 16 and 17.
# Standardize different column naming conventions.
# Rank models using the main fraud-detection metrics.
# ------------------------------------------------------------

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

METRICS_DIR = "outputs/metrics"

baseline_file = os.path.join(
    METRICS_DIR,
    "stage_15_baseline_model_metrics.csv"
)

advanced_file = os.path.join(
    METRICS_DIR,
    "stage_16_advanced_model_metrics.csv"
)

imbalance_file = os.path.join(
    METRICS_DIR,
    "stage_17_imbalance_model_metrics.csv"
)

output_file = os.path.join(
    METRICS_DIR,
    "stage_18_model_comparison.csv"
)

ranking_file = os.path.join(
    METRICS_DIR,
    "stage_18_model_ranking.csv"
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

required_files = [
    baseline_file,
    advanced_file,
    imbalance_file
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Required file not found: {file}"
        )

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

baseline = pd.read_csv(baseline_file)
advanced = pd.read_csv(advanced_file)
imbalance = pd.read_csv(imbalance_file)

# ------------------------------------------------------------
# STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

def standardize_columns(df):

    rename_map = {
        "model": "Model",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "f1_score": "F1",
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "specificity": "Specificity",
        "fpr": "FPR"
    }

    df = df.rename(
        columns=rename_map
    )

    return df


baseline = standardize_columns(baseline)
advanced = standardize_columns(advanced)
imbalance = standardize_columns(imbalance)

# ------------------------------------------------------------
# ADD SOURCE
# ------------------------------------------------------------

baseline["Source"] = "Stage 15"
advanced["Source"] = "Stage 16"
imbalance["Source"] = "Stage 17"

# ------------------------------------------------------------
# COMBINE
# ------------------------------------------------------------

comparison = pd.concat(
    [
        baseline,
        advanced,
        imbalance
    ],
    ignore_index=True
)

# ------------------------------------------------------------
# REQUIRED MODEL COLUMN
# ------------------------------------------------------------

if "Model" not in comparison.columns:
    raise ValueError(
        "Model column was not found after standardization."
    )

# ------------------------------------------------------------
# SELECT CORE METRICS
# ------------------------------------------------------------

preferred_columns = [
    "Model",
    "Precision",
    "Recall",
    "F1",
    "PR-AUC",
    "ROC-AUC",
    "Specificity",
    "FPR",
    "Source"
]

available_columns = [
    column
    for column in preferred_columns
    if column in comparison.columns
]

comparison = comparison[
    available_columns
]

# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

numeric_columns = [
    "Precision",
    "Recall",
    "F1",
    "PR-AUC",
    "ROC-AUC",
    "Specificity",
    "FPR"
]

for column in numeric_columns:

    if column in comparison.columns:

        comparison[column] = pd.to_numeric(
            comparison[column],
            errors="coerce"
        )

# ------------------------------------------------------------
# REMOVE DUPLICATE MODELS
# ------------------------------------------------------------

comparison = comparison.drop_duplicates(
    subset=["Model"],
    keep="last"
)

# ------------------------------------------------------------
# SORT BY PR-AUC
# ------------------------------------------------------------

if "PR-AUC" in comparison.columns:

    comparison = comparison.sort_values(
        by="PR-AUC",
        ascending=False
    )

# ------------------------------------------------------------
# SAVE MODEL COMPARISON
# ------------------------------------------------------------

comparison.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# MODEL RANKING
# ------------------------------------------------------------

ranking = comparison.copy()

# ------------------------------------------------------------
# INDIVIDUAL RANKS
# ------------------------------------------------------------

rank_definitions = {
    "PR-AUC": "PR_AUC_Rank",
    "Recall": "Recall_Rank",
    "Precision": "Precision_Rank",
    "F1": "F1_Rank",
    "ROC-AUC": "ROC_AUC_Rank"
}

for metric, rank_column in rank_definitions.items():

    if metric in ranking.columns:

        ranking[rank_column] = (
            ranking[metric]
            .rank(
                ascending=False,
                method="min"
            )
        )

# ------------------------------------------------------------
# OVERALL RANK
# ------------------------------------------------------------

rank_columns = [
    "PR_AUC_Rank",
    "Recall_Rank",
    "Precision_Rank",
    "F1_Rank",
    "ROC_AUC_Rank"
]

available_rank_columns = [
    column
    for column in rank_columns
    if column in ranking.columns
]

ranking["Overall_Rank"] = (
    ranking[available_rank_columns]
    .mean(axis=1)
)

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

ranking = ranking.sort_values(
    by="Overall_Rank",
    ascending=True
)

# ------------------------------------------------------------
# SAVE RANKING
# ------------------------------------------------------------

ranking.to_csv(
    ranking_file,
    index=False
)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 18 — MODEL COMPARISON")
print("==============================================")

print("\nModel Performance:")
print(
    comparison.to_string(
        index=False
    )
)

print("\nOverall Ranking:")

print(
    ranking[
        [
            "Model",
            "Overall_Rank"
        ]
    ].to_string(
        index=False
    )
)

print("\nBest Candidate:")

print(
    ranking.iloc[0]["Model"]
)

print("\nBest Candidate Metrics:")

best_model = ranking.iloc[0]

for metric in [
    "Precision",
    "Recall",
    "F1",
    "PR-AUC",
    "ROC-AUC"
]:

    if metric in best_model:

        print(
            f"{metric}: "
            f"{best_model[metric]:.6f}"
        )

print("\nOutput files:")
print(output_file)
print(ranking_file)

print("\nSTAGE 18 COMPLETE")