"""
Trains and evaluates fraud-classification models.

Fraud detection is a severe class-imbalance problem (~2% positive class), so
accuracy is a useless metric here (predicting "not fraud" for everyone would
score ~97.6%). This script evaluates on precision, recall, F1, and PR-AUC,
and compares a baseline Logistic Regression against a Random Forest.
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, precision_recall_curve, classification_report
)

df = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

customers = pd.read_csv("data/customers.csv")
df = df.merge(customers[["customer_id", "avg_monthly_spend"]], on="customer_id", how="left")
df["amount_vs_customer_norm"] = df["amount"] / (df["avg_monthly_spend"] / 30.0)
df["is_rapid_repeat"] = (df["seconds_since_prev_txn"] < 120).astype(int)

FEATURES_NUM = ["amount", "seconds_since_prev_txn", "merchant_risk_score",
                 "hour_of_day", "day_of_week", "amount_vs_customer_norm", "is_rapid_repeat"]
FEATURES_CAT = ["category"]
TARGET = "is_fraud"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])

models = {
    "logistic_regression": Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
    "random_forest": Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ]),
}

results = {}
Path("reports").mkdir(exist_ok=True)
plt.figure(figsize=(7, 6))

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    results[name] = {
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, target_names=["legit", "fraud"]),
    }

    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.plot(rec, prec, label=f"{name} (PR-AUC={results[name]['pr_auc']:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Fraud Detection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/precision_recall_curve.png", dpi=150)

# feature importance from the random forest, mapped back to readable names
rf = models["random_forest"].named_steps["clf"]
cat_names = list(models["random_forest"].named_steps["prep"]
                  .named_transformers_["cat"].get_feature_names_out(FEATURES_CAT))
all_feature_names = FEATURES_NUM + cat_names
importances = pd.Series(rf.feature_importances_, index=all_feature_names).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
importances.head(12).sort_values().plot(kind="barh")
plt.title("Random Forest — Top Feature Importances")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=150)

with open("reports/model_metrics.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
               for k, v in results.items()}, f, indent=2)

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("confusion_matrix", "classification_report")}
                   for k, v in results.items()}, indent=2))
for name, r in results.items():
    print(f"\n=== {name} ===")
    print(r["classification_report"])
    print("confusion matrix [[TN FP][FN TP]]:", r["confusion_matrix"])

print("\ntop features (random forest):")
print(importances.head(10))
