# Fraud Detection Engine

[![CI](https://github.com/dhandashreya/fraud-detection-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/dhandashreya/fraud-detection-engine/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An end-to-end fraud-detection pipeline: synthetic transaction data → SQL analysis → a
classification model evaluated the way fraud actually has to be evaluated — on
precision/recall, not accuracy.

## Why synthetic data

Public fraud datasets (e.g. Kaggle's credit-card-fraud set) are PCA-anonymized —
the columns are literally named `V1`...`V28` with no real meaning, which makes for
a weak SQL/EDA story. This project generates its own transaction stream instead,
with interpretable fields and deliberately injected fraud patterns:

- **Odd-hour transactions** (midnight–4am)
- **Amount spikes** relative to a customer's own spending history
- **High-risk categories** (electronics, jewelry, online retail, ATM withdrawals)
- **Card-testing bursts** — several small rapid-fire charges from the same
  customer within seconds of each other, a real-world fraud pattern

`seconds_since_prev_txn` is not a fabricated label — it's computed per customer
from the actual transaction timestamps (`groupby("customer_id")["timestamp"].diff()`),
so both the SQL queries and the model are working off a genuine feature.

## Pipeline

```bash
pip install -r requirements.txt
python src/generate_data.py      # generates data/{customers,merchants,transactions}.csv
python src/load_to_sqlite.py     # loads CSVs into data/fraud.db
python src/run_sql_report.py     # runs sql/analysis_queries.sql -> reports/sql_findings.md
python src/eda.py                # -> reports/eda_*.png
python src/train_model.py        # trains + evaluates models -> reports/*.png, model_metrics.json
```

## Tests

`tests/test_pipeline.py` runs the actual pipeline scripts end to end (not a
reimplementation) and asserts on the real output: dataset scale, card-testing
burst detection, the SQL report's contents, and a minimum performance bar on
the trained model (recall > 0.85, PR-AUC > 0.85) — so a future change that
silently degrades the model fails CI instead of shipping quietly.

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

Runs automatically on every push via [GitHub Actions](.github/workflows/ci.yml).

## Dataset

59,356 transactions · 2,000 customers · 300 merchants · **2.39% fraud rate**
(realistic for card fraud — this is why accuracy is the wrong metric below)

## SQL analysis

Full runnable query set: [`sql/analysis_queries.sql`](sql/analysis_queries.sql) · results: [`reports/sql_findings.md`](reports/sql_findings.md)

Highlights:
- **Category risk is concentrated**: electronics, online retail, ATM withdrawals, and
  jewelry have a 5-6% fraud rate — 15-20x higher than grocery/fuel/dining (~0.3%)
- **73% of fraud happens between midnight and 5am**, vs. a roughly flat rate
  the rest of the day
- A window-function query (`LAG() OVER (PARTITION BY customer_id ORDER BY timestamp)`)
  isolates card-testing bursts: customers with ≥2 transactions under 120 seconds
  apart are fraud in the large majority of cases
- A risk-bucket validation query shows the merchant `merchant_risk_score` field
  (assigned independently of the fraud labels) does **not** actually correlate with
  real fraud rate — a useful negative result showing category and behavior matter
  more than a merchant's static risk score

![Fraud rate by category and hour](reports/eda_category_hour.png)

## Model results

Class imbalance (2.39% positive) makes accuracy meaningless — a model that
predicts "not fraud" for every transaction scores 97.6% accuracy while catching
zero fraud. Evaluated on **precision, recall, F1, and PR-AUC** instead.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.353 | 0.800 | 0.490 | 0.810 |
| **Random Forest (balanced)** | **0.563** | **0.941** | **0.705** | **0.938** |

The Random Forest catches **94% of fraud** while keeping precision at 56% — in a
real deployment this is the recall/precision trade-off a fraud team would tune
based on the cost of a missed fraud vs. the cost of a false alarm (e.g. flagging
for manual review vs. auto-declining).

**Top predictive features**: transaction amount relative to the customer's own
spending norm, raw amount, seconds since the customer's previous transaction, and
the rapid-repeat flag — confirming the model is actually learning the behavioral
fraud signals rather than shortcutting on merchant category.

![Feature importance](reports/feature_importance.png)

## Project structure

```
data/        generated CSVs + SQLite database (db file gitignored, regenerate via scripts)
sql/         schema + analyst queries
src/         pipeline scripts (data gen, load, SQL report, EDA, model training)
reports/     generated charts, SQL findings, model metrics
```

## Stack

Python · pandas · SQLite · scikit-learn (Logistic Regression, Random Forest) ·
matplotlib/seaborn

## License

MIT — see [LICENSE](LICENSE).
