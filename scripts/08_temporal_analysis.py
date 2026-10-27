#Stage 08 — Temporal Analysis

#Purpose: Analyze fraud patterns across time using the hourly step variable and 
# derive temporal indicators such as hour, day, and weekend status.
# scripts/08_temporal_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# PATHS
# ==============================

INPUT_FILE = Path("data/interim/cleaned_transactions.csv")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# TEMPORAL FEATURES
# ==============================

# PaySim step represents an hourly time interval
df["hour"] = (df["step"] - 1) % 24

df["day"] = ((df["step"] - 1) // 24) + 1

df["day_of_week"] = (df["day"] - 1) % 7

df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ==============================
# HOURLY ANALYSIS
# ==============================

hourly_analysis = (
    df.groupby("hour")
    .agg(
        total_transactions=("isFraud", "count"),
        fraudulent_transactions=("isFraud", "sum")
    )
    .reset_index()
)

hourly_analysis["fraud_rate_percentage"] = (
    hourly_analysis["fraudulent_transactions"]
    / hourly_analysis["total_transactions"]
) * 100

hourly_analysis.to_csv(
    TABLE_DIR / "stage_08_hourly_analysis.csv",
    index=False
)

# ==============================
# DAILY ANALYSIS
# ==============================

daily_analysis = (
    df.groupby("day")
    .agg(
        total_transactions=("isFraud", "count"),
        fraudulent_transactions=("isFraud", "sum")
    )
    .reset_index()
)

daily_analysis["fraud_rate_percentage"] = (
    daily_analysis["fraudulent_transactions"]
    / daily_analysis["total_transactions"]
) * 100

daily_analysis.to_csv(
    TABLE_DIR / "stage_08_daily_analysis.csv",
    index=False
)

# ==============================
# DAY OF WEEK ANALYSIS
# ==============================

weekday_analysis = (
    df.groupby("day_of_week")
    .agg(
        total_transactions=("isFraud", "count"),
        fraudulent_transactions=("isFraud", "sum")
    )
    .reset_index()
)

weekday_analysis["fraud_rate_percentage"] = (
    weekday_analysis["fraudulent_transactions"]
    / weekday_analysis["total_transactions"]
) * 100

weekday_analysis.to_csv(
    TABLE_DIR / "stage_08_day_of_week_analysis.csv",
    index=False
)

# ==============================
# WEEKEND VS WEEKDAY
# ==============================

weekend_analysis = (
    df.groupby("is_weekend")
    .agg(
        total_transactions=("isFraud", "count"),
        fraudulent_transactions=("isFraud", "sum")
    )
    .reset_index()
)

weekend_analysis["fraud_rate_percentage"] = (
    weekend_analysis["fraudulent_transactions"]
    / weekend_analysis["total_transactions"]
) * 100

weekend_analysis["period"] = weekend_analysis["is_weekend"].map({
    0: "Weekday",
    1: "Weekend"
})

weekend_analysis.to_csv(
    TABLE_DIR / "stage_08_weekend_analysis.csv",
    index=False
)

# ==============================
# FRAUD BY HOUR
# ==============================

plt.figure(figsize=(12, 6))

plt.plot(
    hourly_analysis["hour"],
    hourly_analysis["fraud_rate_percentage"],
    marker="o"
)

plt.title("Fraud Rate by Hour")
plt.xlabel("Hour")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(24))

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_08_fraud_rate_by_hour.png"
)

plt.close()

# ==============================
# TRANSACTION VOLUME BY HOUR
# ==============================

plt.figure(figsize=(12, 6))

plt.bar(
    hourly_analysis["hour"],
    hourly_analysis["total_transactions"]
)

plt.title("Transaction Volume by Hour")
plt.xlabel("Hour")
plt.ylabel("Number of Transactions")
plt.xticks(range(24))

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_08_transaction_volume_by_hour.png"
)

plt.close()

# ==============================
# FRAUD BY DAY
# ==============================

plt.figure(figsize=(12, 6))

plt.plot(
    daily_analysis["day"],
    daily_analysis["fraudulent_transactions"],
    marker="o"
)

plt.title("Fraudulent Transactions by Day")
plt.xlabel("Day")
plt.ylabel("Number of Fraudulent Transactions")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "stage_08_fraud_by_day.png"
)

plt.close()

# ==============================
# COMPLETE
# ==============================

print("STAGE 08 COMPLETE")
print(f"Time steps analyzed: {df['step'].nunique()}")
print(f"Hours represented: {df['hour'].nunique()}")
print(f"Days represented: {df['day'].nunique()}")
print("Temporal analysis tables saved.")
print("Temporal analysis figures saved.")