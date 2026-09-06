"""
End-to-end pipeline tests: each stage is run as the real script (not
reimplemented), then the actual output artifacts are checked. This is what
the CI workflow runs on every push -- it would catch, for example, a future
edit that broke the injected fraud rate, silently degraded model recall, or
made the trained model unloadable by the scoring script.
"""
import json
import subprocess
import sys

import pandas as pd
import pytest


def run(script, *args):
    result = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module", autouse=True)
def run_pipeline():
    run("src/generate_data.py")
    run("src/load_to_sqlite.py")
    run("src/run_sql_report.py")
    run("src/train_model.py")
    run("src/score.py")
    run("src/explain_model.py")


def test_data_generation_produces_expected_scale():
    df = pd.read_csv("data/transactions.csv")
    assert 55000 <= len(df) <= 62000
    fraud_rate = df["is_fraud"].mean()
    assert 0.01 <= fraud_rate <= 0.04, f"fraud rate {fraud_rate} out of expected range"


def test_data_generation_is_deterministic():
    first = pd.read_csv("data/transactions.csv")
    run("src/generate_data.py")
    second = pd.read_csv("data/transactions.csv")
    pd.testing.assert_frame_equal(first, second)


def test_card_testing_bursts_are_detectable():
    df = pd.read_csv("data/transactions.csv")
    rapid = (df["seconds_since_prev_txn"] < 120).sum()
    assert rapid > 100, "card-testing bursts should produce a meaningful number of rapid-repeat transactions"


def test_sql_findings_report_generated():
    content = open("reports/sql_findings.md", encoding="utf-8").read()
    assert "Headline fraud rate" in content
    assert "Card-testing pattern" in content


def test_model_meets_minimum_performance_bar():
    metrics = json.load(open("reports/model_metrics.json"))
    rf = metrics["random_forest"]
    assert rf["recall"] > 0.85, "Random Forest recall regressed below 0.85"
    assert rf["pr_auc"] > 0.85, "Random Forest PR-AUC regressed below 0.85"


def test_cost_optimal_threshold_is_no_worse_than_default():
    analysis = json.load(open("reports/model_metrics.json"))["threshold_analysis"]
    assert 0.0 < analysis["cost_optimal"]["threshold"] < 1.0
    assert analysis["cost_optimal"]["expected_cost"] <= analysis["default_0.5"]["expected_cost"]


def test_scoring_flags_transactions_and_persists_model():
    from pathlib import Path
    assert Path("models/fraud_model.joblib").exists()

    flagged = pd.read_csv("reports/flagged_transactions.csv")
    assert len(flagged) > 0
    assert flagged["fraud_score"].between(0, 1).all()


def test_shap_explanation_agrees_with_the_model():
    from pathlib import Path
    for f in ("shap_summary.png", "shap_importance.png", "shap_waterfall_fraud.png",
              "shap_findings.md"):
        assert Path("reports", f).exists(), f"explain_model.py did not produce {f}"

    shap_importance = json.load(open("reports/shap_importance.json"))
    top_by_shap = max(shap_importance, key=shap_importance.get)
    # the behavioural amount features should dominate, not merchant category
    assert top_by_shap in ("amount", "amount_vs_customer_norm")
    assert not top_by_shap.startswith("category_")
