#Stage 16 — Advanced Models

#Purpose: Train stronger machine-learning models—Random Forest, 
# Gradient Boosting, XGBoost, and LightGBM—and compare their fraud-detection 
# performance.

# scripts/16_advanced_models.py

# scripts/16_advanced_models.py

import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Optional advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


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
# FRAUD IMBALANCE
# ==============================

fraud_count = y_train.sum()
legitimate_count = len(y_train) - fraud_count

scale_pos_weight = (
    legitimate_count / fraud_count
    if fraud_count > 0
    else 1
)


# ==============================
# MODELS
# ==============================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}


# ==============================
# XGBOOST
# ==============================

if XGBOOST_AVAILABLE:

    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


# ==============================
# LIGHTGBM
# ==============================

if LIGHTGBM_AVAILABLE:

    models["LightGBM"] = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )


# ==============================
# MODEL EVALUATION
# ==============================

results = []

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

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

results_df = results_df.sort_values(
    "pr_auc",
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR / "stage_16_advanced_model_metrics.csv",
    index=False
)


# ==============================
# SAVE FEATURE LIST
# ==============================

pd.DataFrame({
    "feature": features
}).to_csv(
    OUTPUT_DIR / "stage_16_model_features.csv",
    index=False
)


# ==============================
# COMPLETE
# ==============================

print("\nSTAGE 16 COMPLETE")

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
print(
    OUTPUT_DIR /
    "stage_16_advanced_model_metrics.csv"
)