import pandas as pd
import pytest
import os
import tempfile
import io
from unittest.mock import patch
from src.ingest import load_raw, filter_addiction_related, extract_temporal_features, run_eda

CSV_CONTENT_A = (
    ",drugName,condition,review,rating,date,usefulCount\n"
    "0,Drug A,Opiate Dependence,Great drug,9,\"January 1, 2015\",10\n"
    "1,Drug B,Depression,Helped me,7,\"March 5, 2016\",5\n"
)
CSV_CONTENT_B = (
    ",drugName,condition,review,rating,date,usefulCount\n"
    "2,Drug C,Alcohol Use Disorder,Life saver,10,\"June 3, 2017\",20\n"
)

@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "drugsComTrain_raw.csv").write_text(CSV_CONTENT_A)
    (tmp_path / "drugsComTest_raw.csv").write_text(CSV_CONTENT_B)
    return str(tmp_path)

def test_load_raw_merges_train_test(raw_dir):
    config = {"data": {"raw_dir": raw_dir,
                       "train_file": "drugsComTrain_raw.csv",
                       "test_file": "drugsComTest_raw.csv"}}
    df = load_raw(config)
    assert len(df) == 3
    assert set(df.columns) >= {"drugName", "condition", "review", "rating", "date", "usefulCount"}

def test_load_raw_resets_index(raw_dir):
    config = {"data": {"raw_dir": raw_dir,
                       "train_file": "drugsComTrain_raw.csv",
                       "test_file": "drugsComTest_raw.csv"}}
    df = load_raw(config)
    assert list(df.index) == list(range(len(df)))


FILTER_CONFIG = {
    "filter": {
        "addiction_conditions": ["opiate dependence", "alcohol use disorder"],
        "addiction_drugs": ["buprenorphine", "naltrexone"],
    }
}

def make_df(rows):
    return pd.DataFrame(rows, columns=["drugName", "condition", "review", "rating", "date", "usefulCount"])

def test_filter_includes_condition_match():
    df = make_df([
        ["Drug X", "Opiate Dependence", "review text", 8, "Jan 1, 2015", 5],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1
    assert result.iloc[0]["drugName"] == "Drug X"
    assert result.iloc[0]["condition"] == "Opiate Dependence"

def test_filter_includes_drug_match():
    df = make_df([
        ["Buprenorphine", "Back Pain", "review text", 6, "Jan 1, 2015", 2],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1
    assert result.iloc[0]["drugName"] == "Buprenorphine"

def test_filter_excludes_unrelated():
    df = make_df([
        ["Aspirin", "Headache", "review text", 7, "Jan 1, 2015", 1],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 0

def test_filter_deduplicates_union():
    """A row matching both condition AND drug should appear only once."""
    df = make_df([
        ["Buprenorphine", "Opiate Dependence", "review text", 9, "Jan 1, 2015", 3],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1


def test_temporal_features_parse_date():
    df = make_df([
        ["Drug A", "Opiate Dependence", "text", 8, "October 6, 2013", 4],
        ["Drug B", "Alcohol Use Disorder", "text", 5, "March 15, 2016", 2],
    ])
    result = extract_temporal_features(df)
    assert "year" in result.columns
    assert "quarter" in result.columns
    assert "year_quarter" in result.columns
    assert result.iloc[0]["year"] == 2013
    assert result.iloc[0]["quarter"] == 4
    assert result.iloc[0]["year_quarter"] == "2013-Q4"
    assert result.iloc[1]["year_quarter"] == "2016-Q1"


def test_filter_includes_drug_substring_match():
    """Combo drug names like 'Buprenorphine / Naloxone' should match."""
    df = make_df([
        ["Buprenorphine / Naloxone", "Pain Management", "review text", 7, "Jan 1, 2015", 3],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1
    assert result.iloc[0]["drugName"] == "Buprenorphine / Naloxone"


def test_run_eda_does_not_crash():
    df = make_df([
        ["Drug A", "Opiate Dependence", "review text", 8, "October 6, 2013", 4],
        ["Drug B", "Alcohol Use Disorder", "other review", 5, "March 15, 2016", 2],
    ])
    df = extract_temporal_features(df)
    # Should not raise — captures stderr output
    captured = io.StringIO()
    with patch('sys.stderr', captured):
        run_eda(df)
    output = captured.getvalue()
    assert "Total reviews: 2" in output
    assert "Date range:" in output
