# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 24 — TEMPORAL VALIDATION
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Evaluate model performance across chronological time periods.
#
# Earlier transactions are used for training.
# Later transactions are used for validation.
#
# This helps determine whether model performance remains
# stable over time.
# ------------------------------------------------------------


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

TRAIN_FILE = "data/processed/splits/train.csv"
TEST_FILE = "data/processed/splits/test.csv"

OUTPUT_DIR = "outputs/metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_file = os.path.join(
    OUTPUT_DIR,
    "stage_24_temporal_validation.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Required file not found: {TRAIN_FILE}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Required file not found: {TEST_FILE}"
    )

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

TARGET = "isFraud"


# ------------------------------------------------------------
# MODEL FEATURES
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

missing_train = [
    feature
    for feature in FEATURES + [TARGET]
    if feature not in train.columns
]

missing_test = [
    feature
    for feature in FEATURES + [TARGET]
    if feature not in test.columns
]

if missing_train:
    raise ValueError(
        f"Missing columns in training data: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing columns in test data: {missing_test}"
    )


# ------------------------------------------------------------
# COMBINE DATA
# ------------------------------------------------------------

train["Dataset"] = "Train"
test["Dataset"] = "Test"

data = pd.concat(
    [train, test],
    ignore_index=True
)

data = data.sort_values(
    by="step"
).reset_index(drop=True)


# ------------------------------------------------------------
# CREATE TEMPORAL PERIODS
# ------------------------------------------------------------
# Divide the available transaction timeline into
# chronological periods.
#
# The supplied dataset contains steps 1–95.
# Five chronological periods are created.

min_step = int(data["step"].min())
max_step = int(data["step"].max())

period_edges = np.linspace(
    min_step,
    max_step + 1,
    6
)

data["Temporal_Period"] = pd.cut(
    data["step"],
    bins=period_edges,
    labels=[
        "Period_1",
        "Period_2",
        "Period_3",
        "Period_4",
        "Period_5"
    ],
    include_lowest=True,
    right=False
)


# ------------------------------------------------------------
# TEMPORAL MODEL
# ------------------------------------------------------------
# For each period:
#
# Training data = all earlier periods
# Validation data = current period
#
# This prevents future transactions from being used
# to predict earlier transactions.
# ------------------------------------------------------------

periods = [
    "Period_1",
    "Period_2",
    "Period_3",
    "Period_4",
    "Period_5"
]

results = []


for i in range(1, len(periods)):

    train_periods = periods[:i]
    validation_period = periods[i]

    temporal_train = data[
        data["Temporal_Period"].isin(train_periods)
    ].copy()

    temporal_validation = data[
        data["Temporal_Period"] == validation_period
    ].copy()

    if temporal_train.empty:
        continue

    if temporal_validation.empty:
        continue

    X_train = temporal_train[FEATURES].copy()
    y_train = temporal_train[TARGET].astype(int)

    X_validation = temporal_validation[FEATURES].copy()
    y_validation = temporal_validation[TARGET].astype(int)

    # --------------------------------------------------------
    # HANDLE NON-FINITE VALUES
    # --------------------------------------------------------

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    X_validation = X_validation.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # --------------------------------------------------------
    # CHECK CLASS AVAILABILITY
    # --------------------------------------------------------

    if y_train.nunique() < 2:
        print(
            f"\nSkipping {validation_period}: "
            "training data contains only one class."
        )
        continue

    if y_validation.nunique() < 2:
        print(
            f"\nWarning: {validation_period} validation data "
            "contains only one class."
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    if y_validation.nunique() == 2:
        pr_auc = average_precision_score(
            y_validation,
            probabilities
        )

        roc_auc = roc_auc_score(
            y_validation,
            probabilities
        )
    else:
        pr_auc = np.nan
        roc_auc = np.nan

    results.append(
        {
            "Validation_Period": validation_period,
            "Training_Periods": ", ".join(
                train_periods
            ),
            "Training_Transactions": len(
                temporal_train
            ),
            "Validation_Transactions": len(
                temporal_validation
            ),
            "Validation_Fraud": int(
                y_validation.sum()
            ),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "PR_AUC": pr_auc,
            "ROC_AUC": roc_auc
        }
    )


# ------------------------------------------------------------
# RESULTS DATAFRAME
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 24 — TEMPORAL VALIDATION")
print("==============================================")

print(
    f"\nAvailable timeline: "
    f"Step {min_step} → Step {max_step}"
)

print(
    "\nTemporal Validation Results:"
)

if results_df.empty:
    print(
        "No valid temporal validation periods were produced."
    )
else:
    print(
        results_df.to_string(index=False)
    )

    print("\nAverage Temporal Performance:")

    print(
        f"Precision : "
        f"{results_df['Precision'].mean():.4f}"
    )

    print(
        f"Recall    : "
        f"{results_df['Recall'].mean():.4f}"
    )

    print(
        f"F1        : "
        f"{results_df['F1'].mean():.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{results_df['PR_AUC'].mean():.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{results_df['ROC_AUC'].mean():.4f}"
    )


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

print("\nOutput file:")
print(output_file)

print("\nStage 24 completed successfully.")