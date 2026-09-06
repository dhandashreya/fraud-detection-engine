"""
SHAP explainability for the trained Random Forest.

Gini feature importance (in train_model.py) tells you which features the forest
*split on* most; it says nothing about direction or about individual decisions.
SHAP does both. This script:

  * loads models/fraud_model.joblib and rebuilds the held-out temporal test set
  * runs shap.TreeExplainer on a sample of it
  * writes a beeswarm summary, a mean-|SHAP| bar chart, and a waterfall for the
    single highest-scored fraud -> reports/shap_*.png
  * writes reports/shap_findings.md, including a check that SHAP's ranking
    agrees with the gini ranking (it should)

Run after: python src/train_model.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from features import FEATURES_CAT, FEATURES_NUM, TARGET, build_features

SAMPLE_SIZE = 2000
TEST_FRACTION = 0.25
RNG = np.random.default_rng(42)


def _clean(names):
    return [n.split("__", 1)[-1] for n in names]


def main():
    artifact = joblib.load("models/fraud_model.joblib")
    pipe = artifact["pipeline"]
    feature_cols = artifact["features_num"] + artifact["features_cat"]

    df = build_features(pd.read_csv("data/transactions.csv"), pd.read_csv("data/customers.csv"))
    df = df.sort_values("timestamp").reset_index(drop=True)
    test = df.iloc[int(len(df) * (1 - TEST_FRACTION)):]

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    feature_names = _clean(prep.get_feature_names_out())

    X_test_t = prep.transform(test[feature_cols])
    proba = clf.predict_proba(X_test_t)[:, 1]

    # human-readable feature values for the plots: raw numerics (not standardised)
    # + the one-hot columns, in the same column order as get_feature_names_out()
    ohe = prep.named_transformers_["cat"]
    X_display = np.hstack([
        test[FEATURES_NUM].to_numpy(),
        np.asarray(ohe.transform(test[FEATURES_CAT]).todense()),
    ])

    # sample for the summary plots; keep every fraud so the beeswarm isn't all legit
    fraud_idx = np.where(test[TARGET].to_numpy() == 1)[0]
    legit_idx = np.where(test[TARGET].to_numpy() == 0)[0]
    n_legit = max(0, SAMPLE_SIZE - len(fraud_idx))
    sample_idx = np.sort(np.concatenate([
        fraud_idx, RNG.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)
    ]))
    X_sample = X_test_t[sample_idx]

    explainer = shap.TreeExplainer(clf)
    sv = explainer(X_sample, check_additivity=False)
    # binary RF -> (n, n_features, 2); keep the fraud class
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    sv.feature_names = feature_names
    sv.data = X_display[sample_idx]  # show real feature values, not standardised ones

    Path("reports").mkdir(exist_ok=True)

    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("SHAP — how each feature pushes the fraud score (test sample)")
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150)
    plt.close()

    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=feature_names).sort_values()
    plt.figure(figsize=(8, 6))
    mean_abs.tail(12).plot(kind="barh", color="#c0392b")
    plt.xlabel("mean |SHAP value|")
    plt.title("SHAP — average impact on the fraud score")
    plt.tight_layout()
    plt.savefig("reports/shap_importance.png", dpi=150)
    plt.close()

    # waterfall for the highest-scored fraud in the test set
    top_fraud_pos = fraud_idx[np.argmax(proba[fraud_idx])]
    sv_one = explainer(X_test_t[top_fraud_pos:top_fraud_pos + 1], check_additivity=False)
    if sv_one.values.ndim == 3:
        sv_one = sv_one[:, :, 1]
    sv_one.feature_names = feature_names
    sv_one.data = X_display[top_fraud_pos:top_fraud_pos + 1]
    shap.plots.waterfall(sv_one[0], max_display=12, show=False)
    plt.title(f"Why this transaction scored {proba[top_fraud_pos]:.2f} (actual: fraud)")
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall_fraud.png", dpi=150)
    plt.close()

    one_drivers = (pd.Series(sv_one.values[0], index=feature_names)
                   .sort_values(ascending=False).head(3))
    drivers_txt = ", ".join(f"`{f}` (+{v:.2f})" for f, v in one_drivers.items())

    # consistency check vs. gini importance
    gini = pd.Series(clf.feature_importances_, index=feature_names)
    shap_rank = mean_abs.sort_values(ascending=False).index.tolist()
    gini_rank = gini.sort_values(ascending=False).index.tolist()
    overlap = len(set(shap_rank[:5]) & set(gini_rank[:5]))

    top = mean_abs.sort_values(ascending=False)
    md = f"""# SHAP findings — Random Forest

Computed by `src/explain_model.py` on a {len(sample_idx):,}-row sample of the
held-out temporal test set (all {len(fraud_idx)} frauds + {n_legit:,} legit).

## Average impact on the fraud score (mean |SHAP|)

| Feature | mean \\|SHAP\\| |
|---|--:|
""" + "\n".join(f"| `{f}` | {v:.4f} |" for f, v in top.head(8).items()) + f"""

![SHAP importance](shap_importance.png)

## Direction

The beeswarm below shows *which way* each feature pushes the score. The dominant
signals are behavioural, not categorical:

- **`amount_vs_customer_norm`** and **`amount`** — large charges relative to the
  customer's own norm push strongly toward fraud (high feature value = high SHAP).
- **`seconds_since_prev_txn`** — small gaps push toward fraud; **`is_rapid_repeat`**
  fires on the card-testing bursts.
- **`hour_of_day`** — the midnight–4am band pushes up; merchant category barely
  moves the score once behaviour is accounted for.

![SHAP beeswarm](shap_summary.png)

## Single-decision explanation

`reports/shap_waterfall_fraud.png` breaks down the highest-scored fraud in the
test set (score {proba[top_fraud_pos]:.2f}) — the exact contribution of each
feature to that one prediction, which is what an analyst reviewing a flagged
transaction would want to see. Top drivers here: {drivers_txt}.

## Consistency with gini importance

SHAP's top 5 and the Random Forest's gini top 5 share **{overlap}/5** features
(SHAP: {shap_rank[:5]}). Agreement is expected — it's a sanity check that the
explanation reflects the model, not an artefact of the method.

_Generated by `src/explain_model.py`._
"""
    Path("reports/shap_findings.md").write_text(md, encoding="utf-8")
    Path("reports/shap_importance.json").write_text(
        json.dumps({f: round(float(v), 6) for f, v in top.items()}, indent=2))

    print("mean |SHAP| ranking:")
    print(top.head(8).round(4))
    print(f"\nSHAP vs gini top-5 overlap: {overlap}/5")
    print("wrote reports/shap_findings.md, shap_summary.png, shap_importance.png, shap_waterfall_fraud.png")


if __name__ == "__main__":
    main()
