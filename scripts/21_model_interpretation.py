# ============================================================
# FRAUD INTELLIGENCE 001
# STAGE 21 — MODEL INTERPRETATION
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

# ------------------------------------------------------------
# PURPOSE
# ------------------------------------------------------------
# Interpret the selected fraud model by measuring:
#
# 1. Built-in feature importance
# 2. Permutation importance
#
# The analysis helps explain WHY transactions receive
# higher fraud-risk predictions.
# ------------------------------------------------------------

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

TRAIN_FILE = "data/processed/splits/train.csv"
TEST_FILE = "data/processed/splits/test.csv"

OUTPUT_DIR = "outputs/metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

feature_file = os.path.join(
    OUTPUT_DIR,
    "stage_21_feature_importance.csv"
)

permutation_file = os.path.join(
    OUTPUT_DIR,
    "stage_21_permutation_importance.csv"
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

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

missing_features = [
    feature
    for feature in FEATURES
    if feature not in train.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------

X_train = train[FEATURES].copy()
y_train = train[TARGET].astype(int)

X_test = test[FEATURES].copy()
y_test = test[TARGET].astype(int)

# ------------------------------------------------------------
# HANDLE NON-FINITE VALUES
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# TRAIN INTERPRETATION MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# ------------------------------------------------------------
# BUILT-IN FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance["Rank"] = range(
    1,
    len(feature_importance) + 1
)

feature_importance.to_csv(
    feature_file,
    index=False
)

# ------------------------------------------------------------
# PERMUTATION IMPORTANCE
# ------------------------------------------------------------

permutation = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance_Mean": permutation.importances_mean,
    "Importance_STD": permutation.importances_std
})

permutation_df = permutation_df.sort_values(
    by="Importance_Mean",
    ascending=False
)

permutation_df["Rank"] = range(
    1,
    len(permutation_df) + 1
)

permutation_df.to_csv(
    permutation_file,
    index=False
)

# ------------------------------------------------------------
# TOP FEATURES
# ------------------------------------------------------------

print("\n==============================================")
print("STAGE 21 — MODEL INTERPRETATION")
print("==============================================")

print("\nBuilt-in Feature Importance:")
print(
    feature_importance.head(10)
    .to_string(index=False)
)

print("\nPermutation Importance:")
print(
    permutation_df.head(10)
    .to_string(index=False)
)

print("\nOutput files:")
print(feature_file)
print(permutation_file)

print("\nStage 21 completed successfully.")