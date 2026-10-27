#Stage 14 — Statistical Analysis

#Purpose: Statistically test relationships between 
# transaction characteristics and fraud using descriptive 
# statistics, correlation, chi-square tests, point-biserial 
# correlation, effect sizes, and confidence intervals.

# scripts/14_statistical_analysis.py

import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import (
    chi2_contingency,
    pointbiserialr,
    pearsonr,
    spearmanr
)

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path(
    "data/processed/splits/train.csv"
)

TABLE_DIR = Path("outputs/tables")

TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# NUMERIC FEATURES
# ==============================

numeric_features = [
    "amount",
    "step",
    "hour",
    "day",
    "day_of_week",
    "is_weekend",
    "origin_transaction_count",
    "destination_transaction_count",
    "origin_unique_recipients",
    "destination_unique_senders",
    "relationship_transaction_count"
]

numeric_features = [
    feature
    for feature in numeric_features
    if feature in df.columns
]

# ==============================
# CORRELATION ANALYSIS
# ==============================

correlation_results = []

for feature in numeric_features:

    x = df[feature]
    y = df["isFraud"]

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    point_r, point_p = pointbiserialr(y, x)

    correlation_results.append({
        "feature": feature,
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p_value": spearman_p,
        "point_biserial_r": point_r,
        "point_biserial_p_value": point_p
    })

correlation_df = pd.DataFrame(
    correlation_results
)

correlation_df.to_csv(
    TABLE_DIR / "stage_14_correlation_analysis.csv",
    index=False
)

# ==============================
# CHI-SQUARE TESTS
# ==============================

categorical_features = [
    "type",
    "is_weekend"
]

chi_square_results = []

for feature in categorical_features:

    if feature not in df.columns:
        continue

    contingency_table = pd.crosstab(
        df[feature],
        df["isFraud"]
    )

    chi2, p_value, degrees_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    chi_square_results.append({
        "feature": feature,
        "chi_square": chi2,
        "p_value": p_value,
        "degrees_of_freedom": degrees_freedom
    })

chi_square_df = pd.DataFrame(
    chi_square_results
)

chi_square_df.to_csv(
    TABLE_DIR / "stage_14_chi_square_tests.csv",
    index=False
)

# ==============================
# FRAUD GROUP DESCRIPTIVE STATS
# ==============================

descriptive_results = []

for feature in numeric_features:

    legitimate = df.loc[
        df["isFraud"] == 0,
        feature
    ]

    fraud = df.loc[
        df["isFraud"] == 1,
        feature
    ]

    descriptive_results.append({
        "feature": feature,

        "legitimate_mean": legitimate.mean(),
        "fraud_mean": fraud.mean(),

        "legitimate_median": legitimate.median(),
        "fraud_median": fraud.median(),

        "legitimate_std": legitimate.std(),
        "fraud_std": fraud.std()
    })

descriptive_df = pd.DataFrame(
    descriptive_results
)

descriptive_df.to_csv(
    TABLE_DIR / "stage_14_fraud_group_statistics.csv",
    index=False
)

# ==============================
# EFFECT SIZE
# ==============================

effect_results = []

for feature in numeric_features:

    legitimate = df.loc[
        df["isFraud"] == 0,
        feature
    ].dropna()

    fraud = df.loc[
        df["isFraud"] == 1,
        feature
    ].dropna()

    mean_difference = (
        fraud.mean() - legitimate.mean()
    )

    pooled_std = np.sqrt(
        (
            ((len(legitimate) - 1) * legitimate.var())
            +
            ((len(fraud) - 1) * fraud.var())
        )
        /
        (
            len(legitimate)
            + len(fraud)
            - 2
        )
    )

    cohens_d = (
        mean_difference / pooled_std
        if pooled_std != 0
        else np.nan
    )

    effect_results.append({
        "feature": feature,
        "mean_difference": mean_difference,
        "cohens_d": cohens_d
    })

effect_df = pd.DataFrame(
    effect_results
)

effect_df.to_csv(
    TABLE_DIR / "stage_14_effect_sizes.csv",
    index=False
)

# ==============================
# FRAUD RATE BY TRANSACTION TYPE
# ==============================

type_statistics = (
    df.groupby("type")
    .agg(
        transaction_count=("isFraud", "size"),
        fraud_count=("isFraud", "sum")
    )
    .reset_index()
)

type_statistics["fraud_rate"] = (
    type_statistics["fraud_count"]
    / type_statistics["transaction_count"]
)

type_statistics.to_csv(
    TABLE_DIR / "stage_14_transaction_type_statistics.csv",
    index=False
)

# ==============================
# 95% CI FOR OVERALL FRAUD RATE
# ==============================

n = len(df)
fraud_count = df["isFraud"].sum()
fraud_rate = fraud_count / n

standard_error = np.sqrt(
    (fraud_rate * (1 - fraud_rate)) / n
)

ci_lower = fraud_rate - 1.96 * standard_error
ci_upper = fraud_rate + 1.96 * standard_error

confidence_interval = pd.DataFrame({
    "metric": ["Fraud Rate"],
    "estimate": [fraud_rate],
    "ci_95_lower": [max(0, ci_lower)],
    "ci_95_upper": [min(1, ci_upper)]
})

confidence_interval.to_csv(
    TABLE_DIR / "stage_14_fraud_rate_confidence_interval.csv",
    index=False
)

# ==============================
# COMPLETE
# ==============================

print("STAGE 14 COMPLETE")
print("Correlation analysis saved.")
print("Chi-square tests saved.")
print("Effect sizes saved.")
print("Confidence interval saved.")