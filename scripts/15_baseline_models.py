#Stage 15 — Baseline Models

#Purpose: Establish baseline fraud-classification 
# performance using Dummy Classifier, Logistic Regression, and 
# Decision Tree models before introducing advanced machine-learning models.

# scripts/15_baseline_models.py

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

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
# MODELS
# ==============================

models = {

    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
}

# ==============================
# MODEL EVALUATION
# ==============================

results = []

for model_name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

    else:

        probabilities = predictions

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    results.append({

        "model": model_name,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "specificity": specificity,

        "false_positive_rate": false_positive_rate,

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
    OUTPUT_DIR / "stage_15_baseline_model_metrics.csv",
    index=False
)

# ==============================
# SAVE FEATURE LIST
# ==============================

pd.DataFrame({
    "feature": features
}).to_csv(
    OUTPUT_DIR / "stage_15_model_features.csv",
    index=False
)

# ==============================
# COMPLETE
# ==============================

print("STAGE 15 COMPLETE")
print("\nBaseline Model Performance:")
print(
    results_df[
        [
            "model",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:")
print(OUTPUT_DIR / "stage_15_baseline_model_metrics.csv")