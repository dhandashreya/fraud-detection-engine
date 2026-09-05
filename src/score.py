"""
Score transactions with the trained fraud model.

Loads models/fraud_model.joblib (the fitted Random Forest + cost-optimal
decision threshold written by src/train_model.py), rebuilds the same features
used in training, and writes the flagged transactions to
reports/flagged_transactions.csv, highest fraud score first.

    python src/train_model.py                 # once, to produce the model
    python src/score.py                        # scores data/transactions.csv
    python src/score.py path/to/other.csv      # scores any CSV with the same columns

If the input has an `is_fraud` column (the bundled dataset does) the script
also prints recall/precision against those labels as a sanity check.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

from features import build_features

MODEL_PATH = Path("models/fraud_model.joblib")
OUTPUT_PATH = Path("reports/flagged_transactions.csv")


def score(txn_csv: str = "data/transactions.csv") -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found — run `python src/train_model.py` first.")

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    feature_cols = artifact["features_num"] + artifact["features_cat"]

    txns = pd.read_csv(txn_csv)
    customers = pd.read_csv("data/customers.csv")
    feat = build_features(txns, customers)

    feat["fraud_score"] = pipeline.predict_proba(feat[feature_cols])[:, 1].round(4)
    feat["flagged"] = (feat["fraud_score"] >= threshold).astype(int)

    flagged = feat[feat["flagged"] == 1].sort_values("fraud_score", ascending=False)
    cols = [c for c in ["transaction_id", "customer_id", "category", "amount",
                        "timestamp", "fraud_score"] if c in flagged.columns]
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    flagged[cols].to_csv(OUTPUT_PATH, index=False)

    print(f"scored {len(feat):,} transactions at threshold {threshold:.2f}")
    print(f"flagged {len(flagged):,} ({len(flagged) / len(feat):.2%}) for review "
          f"-> {OUTPUT_PATH}")

    if "is_fraud" in feat.columns:
        tp = int(((feat.flagged == 1) & (feat.is_fraud == 1)).sum())
        fp = int(((feat.flagged == 1) & (feat.is_fraud == 0)).sum())
        fn = int(((feat.flagged == 0) & (feat.is_fraud == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"vs. known labels: recall {recall:.3f}, precision {precision:.3f} "
              f"({tp} caught, {fn} missed, {fp} false alarms)")

    return flagged


if __name__ == "__main__":
    score(*sys.argv[1:])
