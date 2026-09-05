"""
Feature engineering shared by training (src/train_model.py) and scoring
(src/score.py) so both paths build the exact same columns from a raw
transactions frame -- a mismatch here is the classic train/serve skew bug.
"""
import pandas as pd

FEATURES_NUM = ["amount", "seconds_since_prev_txn", "merchant_risk_score",
                "hour_of_day", "day_of_week", "amount_vs_customer_norm", "is_rapid_repeat"]
FEATURES_CAT = ["category"]
TARGET = "is_fraud"

RAPID_REPEAT_SECONDS = 120


def build_features(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Return `transactions` with the model's engineered columns added.

    `transactions` must have: customer_id, category, timestamp, amount,
    merchant_risk_score, seconds_since_prev_txn (as produced by
    src/generate_data.py). `customers` supplies avg_monthly_spend.
    """
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df = df.merge(customers[["customer_id", "avg_monthly_spend"]], on="customer_id", how="left")
    # how large is this charge next to the customer's own typical daily spend?
    df["amount_vs_customer_norm"] = df["amount"] / (df["avg_monthly_spend"] / 30.0)
    df["is_rapid_repeat"] = (df["seconds_since_prev_txn"] < RAPID_REPEAT_SECONDS).astype(int)
    return df
