"""
Trains and evaluates fraud-classification models.

Fraud detection is a severe class-imbalance problem (~2% positive class), so
accuracy is a useless metric here (predicting "not fraud" for everyone would
score ~97.6%). This script:

  1. splits the data *temporally* (train on the earlier transactions, test on
     the later ones) rather than randomly -- a random split leaks future
     information and flatters the model;
  2. compares a baseline Logistic Regression against a Random Forest on
     precision, recall, F1, and PR-AUC;
  3. turns the model's probabilities into an actual decision by picking the
     threshold that minimises expected dollar cost, not the arbitrary 0.5;
  4. persists the fitted Random Forest + chosen threshold to
     models/fraud_model.joblib so src/score.py can flag new transactions.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FEATURES_CAT, FEATURES_NUM, TARGET, build_features

# --- cost model -------------------------------------------------------------
# A missed fraud (false negative) costs the transaction amount -- the money is
# gone. A false alarm (false positive) costs a flat manual-review fee. These
# two numbers are the only knobs a real fraud team argues about; everything
# downstream (the decision threshold) falls out of them.
REVIEW_COST_PER_FALSE_POSITIVE = 8.0

TEST_FRACTION = 0.25

df = pd.read_csv("data/transactions.csv")
customers = pd.read_csv("data/customers.csv")
df = build_features(df, customers)

# --- temporal split: earlier transactions train, later ones test ------------
df = df.sort_values("timestamp").reset_index(drop=True)
split_at = int(len(df) * (1 - TEST_FRACTION))
train_df, test_df = df.iloc[:split_at], df.iloc[split_at:]
split_ts = test_df["timestamp"].iloc[0]

X_cols = FEATURES_NUM + FEATURES_CAT
X_train, y_train = train_df[X_cols], train_df[TARGET]
X_test, y_test = test_df[X_cols], test_df[TARGET]
test_amounts = test_df["amount"].to_numpy()

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
            class_weight="balanced", random_state=42, n_jobs=-1,
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
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
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
plt.close()

# --- pick the decision threshold that minimises expected dollar cost --------
rf_proba = models["random_forest"].predict_proba(X_test)[:, 1]
y_test_arr = y_test.to_numpy()


def cost_at(threshold: float) -> dict:
    flagged = rf_proba >= threshold
    fp = int(np.sum(flagged & (y_test_arr == 0)))
    missed = ~flagged & (y_test_arr == 1)
    fn = int(np.sum(missed))
    tp = int(np.sum(flagged & (y_test_arr == 1)))
    dollars_missed = float(np.round(test_amounts[missed].sum(), 2))
    total_cost = round(REVIEW_COST_PER_FALSE_POSITIVE * fp + dollars_missed, 2)
    return {
        "threshold": round(float(threshold), 3),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "false_positives": fp,
        "missed_frauds": fn,
        "dollars_lost_to_missed_fraud": dollars_missed,
        "expected_cost": total_cost,
    }


sweep = [cost_at(t) for t in np.linspace(0.05, 0.95, 91)]
optimal = min(sweep, key=lambda r: r["expected_cost"])
default = cost_at(0.5)

thresholds = [r["threshold"] for r in sweep]
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(thresholds, [r["precision"] for r in sweep], label="precision", color="#2980b9")
ax1.plot(thresholds, [r["recall"] for r in sweep], label="recall", color="#27ae60")
ax1.set_xlabel("Decision threshold")
ax1.set_ylabel("Precision / Recall")
ax1.set_ylim(0, 1.05)
ax2 = ax1.twinx()
ax2.plot(thresholds, [r["expected_cost"] for r in sweep], label="expected cost ($)", color="#c0392b")
ax2.set_ylabel("Expected cost ($)")
ax1.axvline(optimal["threshold"], color="#7f8c8d", linestyle="--")
ax1.annotate(f"cost-optimal = {optimal['threshold']:.2f}",
             (optimal["threshold"], 0.1), xytext=(optimal["threshold"] + 0.03, 0.2),
             color="#7f8c8d")
lines = ax1.get_lines()[:2] + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
ax1.set_title("Decision threshold vs. precision, recall, and expected cost")
plt.tight_layout()
plt.savefig("reports/threshold_analysis.png", dpi=150)
plt.close()

# --- feature importance from the random forest, mapped to readable names ----
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
plt.close()

# --- persist the model + chosen threshold for src/score.py -----------------
Path("models").mkdir(exist_ok=True)
joblib.dump({
    "pipeline": models["random_forest"],
    "threshold": optimal["threshold"],
    "features_num": FEATURES_NUM,
    "features_cat": FEATURES_CAT,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "train_rows": int(len(train_df)),
    "sklearn_pr_auc": results["random_forest"]["pr_auc"],
}, "models/fraud_model.joblib")

metrics_out = {
    "split": f"temporal — train on transactions before {split_ts:%Y-%m-%d %H:%M}, "
             f"test on the {len(test_df):,} after ({TEST_FRACTION:.0%})",
    "test_fraud_rate": round(float(y_test.mean()), 4),
    "cost_model": {
        "false_positive": f"${REVIEW_COST_PER_FALSE_POSITIVE:.0f} manual-review fee",
        "false_negative": "full transaction amount (money lost)",
    },
    "threshold_analysis": {
        "default_0.5": default,
        "cost_optimal": optimal,
        "cost_reduction_vs_default_pct": round(
            100 * (default["expected_cost"] - optimal["expected_cost"]) / default["expected_cost"], 1
        ),
    },
}
for name, r in results.items():
    metrics_out[name] = {k: v for k, v in r.items() if k != "classification_report"}

with open("reports/model_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print(json.dumps({k: metrics_out[k] for k in ("split", "logistic_regression", "random_forest",
                                              "threshold_analysis")}, indent=2))
for name, r in results.items():
    print(f"\n=== {name} (threshold 0.5) ===")
    print(r["classification_report"])
    print("confusion matrix [[TN FP][FN TP]]:", r["confusion_matrix"])

print(f"\ncost-optimal threshold: {optimal['threshold']:.2f}  "
      f"(expected cost ${optimal['expected_cost']:,.0f} vs ${default['expected_cost']:,.0f} at 0.5)")
print("\ntop features (random forest):")
print(importances.head(10))
print("\nsaved models/fraud_model.joblib")
