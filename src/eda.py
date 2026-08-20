"""Exploratory data analysis: generates charts used in the README/report."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
Path("reports").mkdir(exist_ok=True)

df = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
df["hour_of_day"] = df["timestamp"].dt.hour

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fraud_by_cat = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False) * 100
sns.barplot(x=fraud_by_cat.values, y=fraud_by_cat.index, ax=axes[0], color="#c0392b")
axes[0].set_xlabel("Fraud rate (%)")
axes[0].set_title("Fraud Rate by Merchant Category")

fraud_by_hour = df.groupby("hour_of_day")["is_fraud"].mean() * 100
sns.lineplot(x=fraud_by_hour.index, y=fraud_by_hour.values, ax=axes[1], marker="o", color="#c0392b")
axes[1].set_xlabel("Hour of day")
axes[1].set_ylabel("Fraud rate (%)")
axes[1].set_title("Fraud Rate by Hour of Day")

plt.tight_layout()
plt.savefig("reports/eda_category_hour.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x="amount", hue="is_fraud", bins=60, log_scale=(True, False),
             stat="density", common_norm=False, palette={0: "#2980b9", 1: "#c0392b"})
ax.set_title("Transaction Amount Distribution: Legit vs Fraud")
ax.set_xlabel("Amount (log scale)")
plt.tight_layout()
plt.savefig("reports/eda_amount_distribution.png", dpi=150)

print("Saved reports/eda_category_hour.png and reports/eda_amount_distribution.png")
print("\nFraud rate by category (%):\n", fraud_by_cat.round(2))
