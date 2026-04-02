import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from src.temporal import (
    compute_stage_distribution,
    compute_drug_trends,
    detect_spikes,
    generate_spike_narratives,
)

def make_temporal_df():
    return pd.DataFrame({
        "year_quarter": ["2013-Q1"] * 10 + ["2013-Q2"] * 10 + ["2013-Q3"] * 10,
        "stage_name": (["RELAPSE_RISK"] * 7 + ["MAINTENANCE"] * 3) +
                      (["RELAPSE_RISK"] * 4 + ["MAINTENANCE"] * 6) +
                      (["RELAPSE_RISK"] * 5 + ["MAINTENANCE"] * 5),
        "risk_level": (["HIGH"] * 7 + ["LOW"] * 3) +
                      (["HIGH"] * 4 + ["LOW"] * 6) +
                      (["HIGH"] * 5 + ["LOW"] * 5),
        "baseline_risk": (["HIGH"] * 7 + ["LOW"] * 3) +
                         (["HIGH"] * 4 + ["LOW"] * 6) +
                         (["HIGH"] * 5 + ["LOW"] * 5),
        "drugName": ["Suboxone"] * 30,
        "rating": [7.0] * 30,
        "usefulCount": list(range(30)),
        "clean_review": [f"review {i}" for i in range(30)],
    })

def test_stage_distribution_sums_to_100():
    df = make_temporal_df()
    dist = compute_stage_distribution(df)
    row_sums = dist.sum(axis=1)
    for val in row_sums:
        assert abs(val - 100.0) < 0.01

def test_drug_trends_has_required_columns():
    df = make_temporal_df()
    trends = compute_drug_trends(df)
    assert "drugName" in trends.columns
    assert "year_quarter" in trends.columns
    assert "median_rating" in trends.columns
    assert "dominant_stage" in trends.columns

def test_detect_spikes_finds_known_spike():
    """Insert a clear spike at Q5 and verify it is detected."""
    quarters = [f"2012-Q{q}" for q in range(1, 5)] + ["2013-Q1"] + [f"2013-Q{q}" for q in range(2, 5)]
    high_counts = [5, 5, 5, 5, 50, 5, 5, 5]
    df_rows = []
    for q, count in zip(quarters, high_counts):
        for _ in range(count):
            df_rows.append({"year_quarter": q, "baseline_risk": "HIGH", "usefulCount": 1, "clean_review": "relapsed"})
        for _ in range(5):
            df_rows.append({"year_quarter": q, "baseline_risk": "LOW", "usefulCount": 1, "clean_review": "sober"})
    df = pd.DataFrame(df_rows)
    config = {"temporal": {"spike_window": 4, "spike_threshold": 2.0}}
    _, spike_quarters = detect_spikes(df, config)
    assert "2013-Q1" in spike_quarters

def test_generate_spike_narratives_mock():
    df = make_temporal_df()
    config = {"llm": {"model": "llama3.1:8b", "temperature": 0.1, "spike_sample_size": 5}}
    spike_quarters = ["2013-Q1"]

    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "During Q1 2013, relapse language spiked. Reviews described medication access issues. This aligned with national opioid availability changes."

    with patch("src.temporal.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = mock_resp
        narratives = generate_spike_narratives(df, spike_quarters, config)

    assert "2013-Q1" in narratives
    assert len(narratives["2013-Q1"]) > 10
