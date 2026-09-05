"""
Sanity-check the modeling discipline on a *real* fraud dataset.

The rest of this project runs on synthetic data (see the README for why). This
script takes the canonical real-world benchmark -- the ULB credit-card-fraud
dataset: 284,807 European card transactions from September 2013, 492 of them
fraud (0.17%) -- and puts it through the exact same pipeline decisions:

  * a temporal train/test split (earlier transactions train, later ones test)
  * the same two models (balanced Logistic Regression vs. Random Forest)
  * imbalanced metrics only (precision / recall / F1 / PR-AUC)
  * the same dollar-cost threshold sweep ($REVIEW_COST per false positive vs.
    the transaction `Amount` per missed fraud)

The features here are PCA components (V1..V28) with no business meaning, so
there is deliberately no SQL / EDA story -- this is purely: does the approach
still work when the fraud wasn't injected by us? (It does -- see
reports/real_data_benchmark.md.)

Data source, in order of preference:
  1. real_data/creditcard.csv   -- e.g. downloaded from Kaggle; has a `Time`
     column, so the split uses it directly.
  2. OpenML dataset 1597, fetched and cached under real_data/openml_cache/.
     OpenML drops `Time` (it is flagged as a row identifier) but preserves the
     original chronological row order, so the split is by row position -- still
     temporal.

Not committed and not run in the main CI job (the dataset is ~150 MB). Run it
manually, or via the real-data-benchmark GitHub workflow.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REVIEW_COST_PER_FALSE_POSITIVE = 8.0
TEST_FRACTION = 0.25
DATA_DIR = Path("real_data")
LOCAL_CSV = DATA_DIR / "creditcard.csv"


def load_data() -> tuple[pd.DataFrame, str]:
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV).sort_values("Time").reset_index(drop=True)
        return df, f"local {LOCAL_CSV} (split on the Time column)"

    from sklearn.datasets import fetch_openml

    print("real_data/creditcard.csv not found — fetching OpenML dataset 1597 "
          "(~150 MB, cached under real_data/openml_cache/)")
    DATA_DIR.mkdir(exist_ok=True)
    bunch = fetch_openml(data_id=1597, as_frame=True, parser="pandas",
                         data_home=str(DATA_DIR / "openml_cache"))
    df = bunch.frame.copy()
    df["Class"] = df["Class"].astype(int)
    # OpenML preserves the original row order, which is sorted by Time.
    return df, "OpenML 1597 (Time dropped as a row id; split by chronological row order)"


def cost_sweep(proba, y, amounts):
    y = np.asarray(y)
    rows = []
    for t in np.linspace(0.05, 0.95, 91):
        flagged = proba >= t
        fp = int(np.sum(flagged & (y == 0)))
        missed = ~flagged & (y == 1)
        fn = int(np.sum(missed))
        tp = int(np.sum(flagged & (y == 1)))
        dollars_missed = float(np.round(amounts[missed].sum(), 2))
        rows.append({
            "threshold": round(float(t), 3),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            "false_positives": fp,
            "missed_frauds": fn,
            "dollars_lost_to_missed_fraud": dollars_missed,
            "expected_cost": round(REVIEW_COST_PER_FALSE_POSITIVE * fp + dollars_missed, 2),
        })
    return rows


def main():
    df, source = load_data()

    feature_cols = [c for c in df.columns if c != "Class"]
    raw_amount = df["Amount"].to_numpy()
    df = df.assign(Amount=np.log1p(df["Amount"]))

    split_at = int(len(df) * (1 - TEST_FRACTION))
    X_train, y_train = df.iloc[:split_at][feature_cols], df.iloc[:split_at]["Class"]
    X_test, y_test = df.iloc[split_at:][feature_cols], df.iloc[split_at:]["Class"]
    test_amounts = raw_amount[split_at:]

    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
    }

    Path("reports").mkdir(exist_ok=True)
    plt.figure(figsize=(7, 6))
    out = {
        "dataset": "ULB credit-card fraud (284,807 transactions, Sep 2013)",
        "source": source,
        "n_transactions": int(len(df)),
        "fraud_rate": round(float(df["Class"].mean()), 5),
        "split": f"temporal — earlier {split_at:,} train / later {len(df) - split_at:,} test",
        "test_fraud_rate": round(float(y_test.mean()), 5),
        "cost_model": {
            "false_positive": f"${REVIEW_COST_PER_FALSE_POSITIVE:.0f} manual-review fee",
            "false_negative": "transaction Amount (money lost)",
        },
    }

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        out[name] = {
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        prec, rec, _ = precision_recall_curve(y_test, proba)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={out[name]['pr_auc']:.3f})")

        if name == "random_forest":
            sweep = cost_sweep(proba, y_test, test_amounts)
            optimal = min(sweep, key=lambda r: r["expected_cost"])
            default = next(r for r in sweep if r["threshold"] == 0.5)
            out["threshold_analysis"] = {
                "default_0.5": default,
                "cost_optimal": optimal,
                "cost_reduction_vs_default_pct": round(
                    100 * (default["expected_cost"] - optimal["expected_cost"])
                    / default["expected_cost"], 1) if default["expected_cost"] else 0.0,
            }

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — ULB credit-card fraud (real data)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/real_data_pr_curve.png", dpi=150)
    plt.close()

    Path("reports/real_data_benchmark.json").write_text(json.dumps(out, indent=2))

    rf, lr, ta = out["random_forest"], out["logistic_regression"], out["threshold_analysis"]
    md = f"""# Real-data benchmark — ULB credit-card fraud

The same pipeline discipline as the synthetic project, applied to a real dataset
to check the approach isn't just memorising rules this repo wrote.

- **Dataset**: {out['dataset']} — fraud rate {out['fraud_rate']:.3%}
- **Source**: {out['source']}
- **Split**: {out['split']} (fraud rate in the test window: {out['test_fraud_rate']:.3%})

## Model results (temporal split, threshold 0.5)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (balanced) | {lr['precision']} | {lr['recall']} | {lr['f1']} | {lr['pr_auc']} |
| **Random Forest (balanced)** | **{rf['precision']}** | **{rf['recall']}** | **{rf['f1']}** | **{rf['pr_auc']}** |

## Cost-based threshold (Random Forest)

Cost model: ${REVIEW_COST_PER_FALSE_POSITIVE:.0f} per false positive vs. the transaction amount per missed fraud.

| Threshold | Precision | Recall | False positives | Missed fraud $ | Expected cost |
|---|---|---|---|---|---|
| 0.50 (default) | {ta['default_0.5']['precision']} | {ta['default_0.5']['recall']} | {ta['default_0.5']['false_positives']} | ${ta['default_0.5']['dollars_lost_to_missed_fraud']:,.0f} | ${ta['default_0.5']['expected_cost']:,.0f} |
| {ta['cost_optimal']['threshold']:.2f} (cost-optimal) | {ta['cost_optimal']['precision']} | {ta['cost_optimal']['recall']} | {ta['cost_optimal']['false_positives']} | ${ta['cost_optimal']['dollars_lost_to_missed_fraud']:,.0f} | ${ta['cost_optimal']['expected_cost']:,.0f} |

Cost reduction vs. the default 0.5 cutoff: **{ta['cost_reduction_vs_default_pct']}%**.

![PR curve, real data](real_data_pr_curve.png)

_Generated by `src/real_data_benchmark.py`._
"""
    Path("reports/real_data_benchmark.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("source", "n_transactions", "fraud_rate",
                                          "logistic_regression", "random_forest",
                                          "threshold_analysis")}, indent=2))
    print("\nwrote reports/real_data_benchmark.md and reports/real_data_pr_curve.png")


if __name__ == "__main__":
    main()
