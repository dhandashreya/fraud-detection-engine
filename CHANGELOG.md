# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0]

### Added
- `src/score.py` — loads the persisted model + threshold and flags transactions
  to `reports/flagged_transactions.csv`.
- `src/features.py` — feature engineering shared by training and scoring so the
  two paths can't drift (train/serve skew).
- Cost-based threshold selection in `src/train_model.py`: sweeps every threshold
  against an explicit `$8 per false positive` vs. `transaction amount per missed
  fraud` cost model and picks the minimum-cost cutoff (`reports/threshold_analysis.png`).
- Model persistence to `models/fraud_model.joblib` (pipeline + chosen threshold).
- Tests for determinism, the cost-optimal threshold, and model load/score.
- `CHANGELOG.md`; "Scoring" and "Limitations" sections in the README.

### Changed
- Train/test split is now **temporal** (earlier transactions train, later ones
  test) instead of random, to remove look-ahead leakage.
- `src/generate_data.py` is now fully deterministic — the two `DataFrame.sample`
  calls were using the pandas global RNG instead of the seeded generator, so the
  dataset changed slightly between runs. Regenerated all data and reports.
- `requirements.txt` pinned to exact versions; added `joblib`.
- `models/` added to `.gitignore` (regenerable ~4 MB artifact).

## [0.1.0]

Initial release: synthetic data generator, SQLite schema + analyst queries with a
generated findings report, EDA charts, Logistic Regression vs. Random Forest
evaluated on precision/recall/PR-AUC, end-to-end pytest suite, and GitHub Actions CI.
