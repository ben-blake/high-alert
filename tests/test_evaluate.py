import pandas as pd
import pytest
from src.evaluate import compute_metrics, compare_approaches, save_comparison


def test_compute_metrics_perfect_prediction():
    y_true = ["HIGH", "LOW", "MODERATE", "HIGH"]
    y_pred = ["HIGH", "LOW", "MODERATE", "HIGH"]
    metrics = compute_metrics(y_true, y_pred, labels=["HIGH", "LOW", "MODERATE"])
    assert metrics["HIGH"]["f1-score"] == pytest.approx(1.0)
    assert metrics["LOW"]["f1-score"] == pytest.approx(1.0)


def test_compute_metrics_partial():
    y_true = ["HIGH", "HIGH", "LOW", "MODERATE"]
    y_pred = ["HIGH", "LOW", "LOW", "MODERATE"]
    metrics = compute_metrics(y_true, y_pred, labels=["HIGH", "LOW", "MODERATE"])
    assert 0.0 <= metrics["HIGH"]["f1-score"] <= 1.0


def test_compare_approaches_has_required_columns():
    df = pd.DataFrame({
        "llm_risk":      ["HIGH", "LOW", "MODERATE", "HIGH"],
        "baseline_risk": ["HIGH", "LOW", "HIGH",     "HIGH"],
        "risk_level":    ["HIGH", "LOW", "MODERATE", "LOW"],
    })
    result = compare_approaches(df)
    assert "approach" in result.columns
    assert set(result["approach"]) == {"baseline_vs_llm", "cluster_vs_llm"}
    assert "macro_f1" in result.columns
