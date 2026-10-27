# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 25 — STRESS TESTING
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Evaluate model robustness under stressed conditions:
#
# 1. Increased fraud prevalence
# 2. Reduced fraud prevalence
# 3. Increased transaction amounts
# 4. Reduced transaction amounts
#
# The model is trained on the original training data and
# evaluated under each stress scenario.
# ------------------------------------------------------------


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

TRAIN_FILE = "data/processed/splits/train.csv"
TEST_FILE = "data/processed/splits/test.csv"

OUTPUT_DIR = "outputs/metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "stage_25_stress_testing.csv"
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

required_columns = FEATURES + [TARGET]

missing_train = [
    column
    for column in required_columns
    if column not in train.columns
]

missing_test = [
    column
    for column in required_columns
    if column not in test.columns
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
# PREPARE TRAINING DATA
# ------------------------------------------------------------

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# BASE TEST DATA
# ------------------------------------------------------------

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


# ------------------------------------------------------------
# BASELINE PREDICTIONS
# ------------------------------------------------------------

baseline_probability = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# STRESS SCENARIO FUNCTION
# ------------------------------------------------------------

def evaluate_scenario(
    scenario_name,
    scenario_data,
    scenario_target
):

    scenario_data = scenario_data.copy()

    scenario_data = scenario_data.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    probabilities = model.predict_proba(
        scenario_data[FEATURES]
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        scenario_target,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        scenario_target,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        scenario_target,
        predictions,
        zero_division=0
    )

    if scenario_target.nunique() == 2:

        pr_auc = average_precision_score(
            scenario_target,
            probabilities
        )

        roc_auc = roc_auc_score(
            scenario_target,
            probabilities
        )

    else:

        pr_auc = np.nan
        roc_auc = np.nan

    return {
        "Scenario": scenario_name,
        "Transactions": len(
            scenario_data
        ),
        "Fraud_Rate": scenario_target.mean(),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    }


# ------------------------------------------------------------
# SCENARIO 1 — BASELINE
# ------------------------------------------------------------

results = []

results.append(
    evaluate_scenario(
        "Baseline",
        X_test,
        y_test
    )
)


# ------------------------------------------------------------
# SCENARIO 2 — INCREASED FRAUD PREVALENCE
# ------------------------------------------------------------
# Duplicate fraud transactions to increase the proportion
# of fraudulent transactions.

fraud_test = test[
    test[TARGET] == 1
].copy()

legitimate_test = test[
    test[TARGET] == 0
].copy()

fraud_stressed = pd.concat(
    [
        fraud_test,
        fraud_test
    ],
    ignore_index=True
)

increased_fraud = pd.concat(
    [
        legitimate_test,
        fraud_stressed
    ],
    ignore_index=True
)

results.append(
    evaluate_scenario(
        "Increased_Fraud_Prevalence",
        increased_fraud[FEATURES],
        increased_fraud[TARGET].astype(int)
    )
)


# ------------------------------------------------------------
# SCENARIO 3 — REDUCED FRAUD PREVALENCE
# ------------------------------------------------------------
# Retain a smaller subset of fraud transactions.

reduced_fraud = fraud_test.sample(
    n=max(
        1,
        int(len(fraud_test) * 0.50)
    ),
    random_state=42
)

reduced_fraud_test = pd.concat(
    [
        legitimate_test,
        reduced_fraud
    ],
    ignore_index=True
)

results.append(
    evaluate_scenario(
        "Reduced_Fraud_Prevalence",
        reduced_fraud_test[FEATURES],
        reduced_fraud_test[TARGET].astype(int)
    )
)


# ------------------------------------------------------------
# SCENARIO 4 — INCREASED TRANSACTION AMOUNTS
# ------------------------------------------------------------

increased_amount = X_test.copy()

increased_amount["amount"] = (
    increased_amount["amount"] * 2
)

increased_amount["amount_log"] = np.log1p(
    increased_amount["amount"]
)

results.append(
    evaluate_scenario(
        "Increased_Transaction_Amounts",
        increased_amount,
        y_test
    )
)


# ------------------------------------------------------------
# SCENARIO 5 — REDUCED TRANSACTION AMOUNTS
# ------------------------------------------------------------

reduced_amount = X_test.copy()

reduced_amount["amount"] = (
    reduced_amount["amount"] * 0.50
)

reduced_amount["amount_log"] = np.log1p(
    reduced_amount["amount"]
)

results.append(
    evaluate_scenario(
        "Reduced_Transaction_Amounts",
        reduced_amount,
        y_test
    )
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 25 — STRESS TESTING")
print("==============================================")

print("\nStress Testing Results:")

print(
    results_df.to_string(index=False)
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nStage 25 completed successfully.")