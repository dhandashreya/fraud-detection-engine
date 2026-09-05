"""
Opt-in test for the real-data benchmark (src/real_data_benchmark.py).

Skipped by default -- it downloads the ~150 MB ULB credit-card-fraud dataset.
The `real-data-benchmark` GitHub workflow runs it with RUN_REAL_DATA=1 and a
cached copy of the dataset; run it locally the same way:

    RUN_REAL_DATA=1 pytest tests/test_real_data_benchmark.py -v
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DATA") != "1",
    reason="set RUN_REAL_DATA=1 to run the ~150 MB real-data benchmark",
)


def test_real_data_benchmark_runs_and_holds_a_performance_bar():
    result = subprocess.run([sys.executable, "src/real_data_benchmark.py"],
                            capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    out = json.loads(Path("reports/real_data_benchmark.json").read_text())
    rf = out["random_forest"]
    assert rf["pr_auc"] > 0.6, "Random Forest PR-AUC on real data regressed below 0.6"
    assert rf["recall"] > 0.5, "Random Forest recall on real data regressed below 0.5"

    optimal = out["threshold_analysis"]["cost_optimal"]
    assert 0.0 < optimal["threshold"] < 1.0
    assert optimal["expected_cost"] <= out["threshold_analysis"]["default_0.5"]["expected_cost"]
