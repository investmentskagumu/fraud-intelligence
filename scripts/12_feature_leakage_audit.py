#Stage 12 — Feature & Leakage Audit

#Purpose: Determine whether each engineered feature is safe for 
# predictive modeling, identify target leakage, future-information leakage, 
# restricted variables, and features requiring exclusion.

# scripts/12_feature_leakage_audit.py

import pandas as pd
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/processed/engineered_transactions.csv")
OUTPUT_FILE = Path(
    "outputs/tables/stage_12_feature_leakage_audit.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# RESTRICTED FEATURES
# ==============================

target_features = [
    "isFraud"
]

restricted_balance_features = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

identifier_features = [
    "nameOrig",
    "nameDest"
]

# ==============================
# POTENTIAL LEAKAGE FEATURES
# ==============================

post_outcome_features = [
    "isFlaggedFraud",
    "flagged_transaction"
]

# ==============================
# FEATURE CLASSIFICATION
# ==============================

audit_results = []

for feature in df.columns:

    if feature in target_features:

        status = "CONFIRMED LEAKAGE"
        reason = "Target variable"

    elif feature in restricted_balance_features:

        status = "EXCLUDED"
        reason = "Restricted balance variable"

    elif feature in post_outcome_features:

        status = "POTENTIAL LEAKAGE"
        reason = "Rule-based indicator requires prediction-time validation"

    elif feature in identifier_features:

        status = "EXCLUDED"
        reason = "Account identifier; not used directly as predictor"

    elif feature in [
        "origin_transaction_count",
        "destination_transaction_count",
        "origin_unique_recipients",
        "destination_unique_senders",
        "relationship_transaction_count"
    ]:

        status = "POTENTIAL LEAKAGE"
        reason = "Behavioral feature requires prediction-time availability validation"

    else:

        status = "SAFE CANDIDATE"
        reason = "Candidate feature requiring final prediction-time validation"

    audit_results.append({
        "feature": feature,
        "status": status,
        "reason": reason
    })

# ==============================
# CREATE AUDIT TABLE
# ==============================

audit_df = pd.DataFrame(audit_results)

# ==============================
# CORRELATION WITH TARGET
# ==============================

numeric_features = df.select_dtypes(
    include="number"
).columns

correlation_data = []

for feature in numeric_features:

    if feature == "isFraud":
        continue

    correlation = df[feature].corr(
        df["isFraud"]
    )

    correlation_data.append({
        "feature": feature,
        "fraud_correlation": correlation
    })

correlation_df = pd.DataFrame(
    correlation_data
)

audit_df = audit_df.merge(
    correlation_df,
    on="feature",
    how="left"
)

# ==============================
# SAVE AUDIT
# ==============================

audit_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==============================
# SUMMARY
# ==============================

summary = (
    audit_df["status"]
    .value_counts()
    .reset_index()
)

summary.columns = [
    "status",
    "feature_count"
]

summary.to_csv(
    "outputs/tables/stage_12_leakage_summary.csv",
    index=False
)

# ==============================
# PRINT RESULTS
# ==============================

print("STAGE 12 COMPLETE")
print("\nFeature Leakage Summary:")
print(summary.to_string(index=False))

print("\nAudit file:")
print(OUTPUT_FILE)