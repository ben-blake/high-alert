import pandas as pd
import pytest
from src.baseline import classify_risk_baseline, batch_classify_baseline

CONFIG = {
    "baseline": {
        "high_risk_patterns": ["relapsed", "relapse", "overdose", "hopeless"],
        "low_risk_patterns": ["sober", "sobriety", "changed my life", "in recovery"],
    }
}

def test_high_risk_detection():
    assert classify_risk_baseline("i relapsed after three months", CONFIG) == "HIGH"

def test_low_risk_detection():
    assert classify_risk_baseline("i have been sober for six months", CONFIG) == "LOW"

def test_moderate_default():
    assert classify_risk_baseline("the medication made me nauseous", CONFIG) == "MODERATE"

def test_high_takes_priority_over_low():
    assert classify_risk_baseline("relapsed but now in recovery", CONFIG) == "HIGH"

def test_batch_classify_adds_column():
    df = pd.DataFrame({"clean_review": [
        "i relapsed last week",
        "sober for a year now",
        "just a regular review",
    ]})
    result = batch_classify_baseline(df, CONFIG)
    assert "baseline_risk" in result.columns
    assert result.iloc[0]["baseline_risk"] == "HIGH"
    assert result.iloc[1]["baseline_risk"] == "LOW"
    assert result.iloc[2]["baseline_risk"] == "MODERATE"
