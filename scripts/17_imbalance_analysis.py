#Stage 17 — Class Imbalance Analysis

#Purpose: Quantify the extreme fraud-class imbalance and 
# evaluate class weighting, threshold-independent performance, and 
# resampling options without relying on accuracy.

# scripts/17_imbalance_analysis.py

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ==============================
# PATHS
# ==============================

TRAIN_FILE = Path(
    "data/processed/splits/train.csv"
)

TEST_FILE = Path(
    "data/processed/splits/test.csv"
)

OUTPUT_DIR = Path("outputs/metrics")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================
# LOAD DATA
# ==============================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


# ==============================
# FEATURES
# ==============================

features = [
    "step",
    "amount",
    "hour",
    "day",
    "day_of_week",
    "is_weekend",
    "is_cash_out",
    "is_transfer",
    "is_payment",
    "is_cash_in",
    "is_debit",
    "amount_log",
    "amount_zero",
    "origin_transaction_count",
    "destination_transaction_count",
    "origin_unique_recipients",
    "destination_unique_senders",
    "relationship_transaction_count"
]

features = [
    feature
    for feature in features
    if feature in train_df.columns
]

X_train = train_df[features]
y_train = train_df["isFraud"]

X_test = test_df[features]
y_test = test_df["isFraud"]


# ==============================
# CLASS DISTRIBUTION
# ==============================

class_counts = (
    y_train
    .value_counts()
    .sort_index()
)

legitimate_count = class_counts.get(0, 0)
fraud_count = class_counts.get(1, 0)

total = len(y_train)

imbalance_ratio = (
    legitimate_count / fraud_count
    if fraud_count > 0
    else np.nan
)

imbalance_summary = pd.DataFrame({
    "class": [0, 1],
    "class_name": [
        "Legitimate",
        "Fraud"
    ],
    "count": [
        legitimate_count,
        fraud_count
    ],
    "percentage": [
        legitimate_count / total * 100,
        fraud_count / total * 100
    ]
})

imbalance_summary.to_csv(
    OUTPUT_DIR / "stage_17_class_distribution.csv",
    index=False
)


# ==============================
# CLASS WEIGHT
# ==============================

balanced_weight = (
    legitimate_count / fraud_count
    if fraud_count > 0
    else 1
)

weight_summary = pd.DataFrame({
    "metric": [
        "Legitimate transactions",
        "Fraudulent transactions",
        "Total transactions",
        "Imbalance ratio",
        "Fraud class weight"
    ],
    "value": [
        legitimate_count,
        fraud_count,
        total,
        imbalance_ratio,
        balanced_weight
    ]
})

weight_summary.to_csv(
    OUTPUT_DIR / "stage_17_imbalance_summary.csv",
    index=False
)


# ==============================
# MODELS
# ==============================

models = {

    "Logistic Regression - Balanced":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Random Forest - Balanced":
        RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
}


# ==============================
# EVALUATION
# ==============================

results = []

for model_name, model in models.items():

    print(f"Training: {model_name}")

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    results.append({
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    })


# ==============================
# SAVE RESULTS
# ==============================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_DIR /
    "stage_17_imbalance_model_metrics.csv",
    index=False
)


# ==============================
# COMPLETE
# ==============================

print("\nSTAGE 17 COMPLETE")
print(f"Legitimate: {legitimate_count:,}")
print(f"Fraud: {fraud_count:,}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

print("\nModel results:")
print(
    results_df[
        [
            "model",
            "precision",
            "recall",
            "f1_score",
            "pr_auc",
            "roc_auc"
        ]
    ].to_string(index=False)
)