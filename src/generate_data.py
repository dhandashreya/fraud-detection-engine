"""
Generates a synthetic credit-card-transaction dataset with realistic fraud patterns.

Real fraud datasets (e.g. Kaggle's PCA-anonymized credit card set) can't be
redistributed with meaningful feature names, so this project generates its own
transaction stream with interpretable features and injected fraud signals:
odd hours, high amounts relative to a customer's history, new/rare merchants,
and rapid repeat transactions ("card testing" bursts).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_CUSTOMERS = 2000
N_MERCHANTS = 300
N_BASE_TRANSACTIONS = 58500
N_CARD_TESTING_BURSTS = 250  # each burst = 2-5 rapid-fire fraud txns for one customer

CATEGORIES = ["grocery", "electronics", "travel", "dining", "fuel",
              "online_retail", "entertainment", "utilities", "jewelry", "atm_withdrawal"]
HIGH_RISK_CATEGORIES = {"electronics", "jewelry", "online_retail", "atm_withdrawal"}

customers = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "home_city": rng.choice(["Toronto", "Vancouver", "Calgary", "Edmonton", "Montreal", "Ottawa"], N_CUSTOMERS),
    "avg_monthly_spend": rng.gamma(shape=3.0, scale=250, size=N_CUSTOMERS).round(2),
    "account_age_days": rng.integers(30, 3650, N_CUSTOMERS),
})

merchants = pd.DataFrame({
    "merchant_id": np.arange(1, N_MERCHANTS + 1),
    "category": rng.choice(CATEGORIES, N_MERCHANTS),
    "merchant_risk_score": rng.beta(2, 8, N_MERCHANTS).round(3),  # most low risk, few high
})

start = datetime(2025, 1, 1)
FRAUD_RATE = 0.009  # baseline scattered fraud; card-testing bursts add more on top

rows = []
for i in range(N_BASE_TRANSACTIONS):
    cust = customers.iloc[rng.integers(0, N_CUSTOMERS)]
    merch = merchants.iloc[rng.integers(0, N_MERCHANTS)]
    ts = start + timedelta(seconds=int(rng.integers(0, 300 * 24 * 3600)))
    is_fraud = rng.random() < FRAUD_RATE

    if is_fraud:
        ts = ts.replace(hour=int(rng.integers(0, 5)))
        amount = round(max(5, cust["avg_monthly_spend"] * rng.uniform(0.8, 4.5) / rng.integers(1, 3)), 2)
        if rng.random() < 0.6:
            merch = merchants[merchants["category"].isin(HIGH_RISK_CATEGORIES)].sample(1).iloc[0]
    else:
        amount = round(max(1, rng.gamma(2.0, cust["avg_monthly_spend"] / 20)), 2)

    rows.append({
        "customer_id": int(cust["customer_id"]),
        "merchant_id": int(merch["merchant_id"]),
        "category": merch["category"],
        "timestamp": ts,
        "amount": amount,
        "merchant_risk_score": float(merch["merchant_risk_score"]),
        "is_fraud": int(is_fraud),
    })

# --- inject explicit card-testing bursts: several small-amount, rapid-fire
# transactions from the same customer within a 20-90 second window ---
for _ in range(N_CARD_TESTING_BURSTS):
    cust = customers.iloc[rng.integers(0, N_CUSTOMERS)]
    burst_len = int(rng.integers(2, 6))
    burst_start = start + timedelta(seconds=int(rng.integers(0, 300 * 24 * 3600)))
    for _ in range(burst_len):
        merch = merchants[merchants["category"].isin(HIGH_RISK_CATEGORIES)].sample(1).iloc[0]
        ts = burst_start + timedelta(seconds=int(rng.integers(5, 90)))
        burst_start = ts
        rows.append({
            "customer_id": int(cust["customer_id"]),
            "merchant_id": int(merch["merchant_id"]),
            "category": merch["category"],
            "timestamp": ts,
            "amount": round(float(rng.uniform(1, 15)), 2),  # small "test" charges
            "merchant_risk_score": float(merch["merchant_risk_score"]),
            "is_fraud": 1,
        })

transactions = pd.DataFrame(rows).sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

# real engineered feature: seconds since this customer's own previous transaction
transactions["seconds_since_prev_txn"] = (
    transactions.groupby("customer_id")["timestamp"].diff().dt.total_seconds()
)
transactions["seconds_since_prev_txn"] = transactions["seconds_since_prev_txn"].fillna(999999).astype(int)

transactions = transactions.sort_values("timestamp").reset_index(drop=True)
transactions.insert(0, "transaction_id", np.arange(1, len(transactions) + 1))

customers.to_csv("data/customers.csv", index=False)
merchants.to_csv("data/merchants.csv", index=False)
transactions.to_csv("data/transactions.csv", index=False)

print(f"customers: {len(customers)}, merchants: {len(merchants)}, transactions: {len(transactions)}")
print(f"fraud rate: {transactions['is_fraud'].mean():.4%}  ({transactions['is_fraud'].sum()} fraudulent txns)")
print(f"txns with <120s gap from customer's own previous txn: {(transactions['seconds_since_prev_txn'] < 120).sum()}")
