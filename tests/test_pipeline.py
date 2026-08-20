"""
End-to-end pipeline tests: each stage is run as the real script (not
reimplemented), then the actual output artifacts are checked. This is what
the CI workflow runs on every push -- it would catch, for example, a future
edit that broke the injected fraud rate or silently degraded model recall.
"""
import json
import subprocess
import sys
import pandas as pd
import pytest


def run(script):
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="module", autouse=True)
def run_pipeline():
    run("src/generate_data.py")
    run("src/load_to_sqlite.py")
    run("src/run_sql_report.py")
    run("src/train_model.py")


def test_data_generation_produces_expected_scale():
    df = pd.read_csv("data/transactions.csv")
    assert 55000 <= len(df) <= 62000
    fraud_rate = df["is_fraud"].mean()
    assert 0.01 <= fraud_rate <= 0.04, f"fraud rate {fraud_rate} out of expected range"


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
