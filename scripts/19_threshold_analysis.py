# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 19 — THRESHOLD ANALYSIS
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PURPOSE
# ============================================================
# Generate fraud probabilities from the selected model and
# evaluate performance across multiple decision thresholds.
#
# The prediction file created here will also be used by:
#
# Stage 20 — Cost & Decision Analysis
# Stage 22 — Risk Scoring
# Stage 23 — Investigation Prioritization
# ============================================================


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "data/processed/splits/train.csv"
TEST_FILE = "data/processed/splits/test.csv"

PREDICTIONS_DIR = "outputs/predictions"
METRICS_DIR = "outputs/metrics"

os.makedirs(
    PREDICTIONS_DIR,
    exist_ok=True
)

os.makedirs(
    METRICS_DIR,
    exist_ok=True
)


# ============================================================
# VALIDATION
# ============================================================

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Test file not found: {TEST_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)


# ============================================================
# TARGET
# ============================================================

TARGET = "isFraud"


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# FEATURE VALIDATION
# ============================================================

missing_train = [
    feature
    for feature in FEATURES
    if feature not in train.columns
]

missing_test = [
    feature
    for feature in FEATURES
    if feature not in test.columns
]

if missing_train:
    raise ValueError(
        f"Missing features in training data: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing features in test data: {missing_test}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int)


# ============================================================
# HANDLE NON-FINITE VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# SELECTED MODEL
# ============================================================

print(
    "Training: Logistic Regression - Balanced"
)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# GENERATE FRAUD PROBABILITIES
# ============================================================

fraud_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# CREATE PREDICTION DATASET
# ============================================================

predictions = test.copy()

predictions["fraud_probability"] = (
    fraud_probability
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    PREDICTIONS_DIR,
    "stage_19_logistic_predictions.csv"
)

predictions.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# THRESHOLD RANGE
# ============================================================

thresholds = np.arange(
    0.01,
    1.00,
    0.01
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

for threshold in thresholds:

    y_pred = (
        fraud_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    results.append({

        "Threshold": threshold,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Specificity": specificity,

        "FPR": fpr,

        "TP": tp,

        "FP": fp,

        "TN": tn,

        "FN": fn

    })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

threshold_results = pd.DataFrame(
    results
)


# ============================================================
# SAVE COMPLETE THRESHOLD ANALYSIS
# ============================================================

threshold_file = os.path.join(
    METRICS_DIR,
    "stage_19_threshold_analysis.csv"
)

threshold_results.to_csv(
    threshold_file,
    index=False
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = threshold_results.loc[
    threshold_results["F1"].idxmax()
]


# ============================================================
# BEST PRECISION
# ============================================================

best_precision = threshold_results.loc[
    threshold_results["Precision"].idxmax()
]


# ============================================================
# BEST RECALL
# ============================================================

best_recall = threshold_results.loc[
    threshold_results["Recall"].idxmax()
]


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([

    {
        "Selection_Criterion": "Best F1",
        "Threshold": best_f1["Threshold"],
        "Precision": best_f1["Precision"],
        "Recall": best_f1["Recall"],
        "F1": best_f1["F1"],
        "Specificity": best_f1["Specificity"],
        "FPR": best_f1["FPR"],
        "TP": best_f1["TP"],
        "FP": best_f1["FP"],
        "TN": best_f1["TN"],
        "FN": best_f1["FN"]
    },

    {
        "Selection_Criterion": "Best Precision",
        "Threshold": best_precision["Threshold"],
        "Precision": best_precision["Precision"],
        "Recall": best_precision["Recall"],
        "F1": best_precision["F1"],
        "Specificity": best_precision["Specificity"],
        "FPR": best_precision["FPR"],
        "TP": best_precision["TP"],
        "FP": best_precision["FP"],
        "TN": best_precision["TN"],
        "FN": best_precision["FN"]
    },

    {
        "Selection_Criterion": "Best Recall",
        "Threshold": best_recall["Threshold"],
        "Precision": best_recall["Precision"],
        "Recall": best_recall["Recall"],
        "F1": best_recall["F1"],
        "Specificity": best_recall["Specificity"],
        "FPR": best_recall["FPR"],
        "TP": best_recall["TP"],
        "FP": best_recall["FP"],
        "TN": best_recall["TN"],
        "FN": best_recall["FN"]
    }

])


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    METRICS_DIR,
    "stage_19_threshold_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("STAGE 19 — THRESHOLD ANALYSIS")
print("==============================================")

print(
    f"\nTest transactions: {len(test):,}"
)

print(
    f"Fraud transactions: {y_test.sum():,}"
)

print(
    f"Legitimate transactions: "
    f"{(y_test == 0).sum():,}"
)

print(
    "\nPrediction file created:"
)

print(prediction_file)


print("\nBest F1 Threshold:")

print(
    best_f1.to_string()
)


print("\nBest Precision Threshold:")

print(
    best_precision.to_string()
)


print("\nBest Recall Threshold:")

print(
    best_recall.to_string()
)


print("\nOutput files:")

print(prediction_file)

print(threshold_file)

print(summary_file)


print("\nSTAGE 19 COMPLETE")