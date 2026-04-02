import pandas as pd
import pytest
import os
import tempfile
from src.ingest import load_raw, filter_addiction_related, extract_temporal_features

TSV_CONTENT_A = (
    "\tdrugName\tcondition\treview\trating\tdate\tusefulCount\n"
    "0\tDrug A\tOpiate Dependence\t\"Great drug\"\t9\tJanuary 1, 2015\t10\n"
    "1\tDrug B\tDepression\t\"Helped me\"\t7\tMarch 5, 2016\t5\n"
)
TSV_CONTENT_B = (
    "\tdrugName\tcondition\treview\trating\tdate\tusefulCount\n"
    "2\tDrug C\tAlcohol Use Disorder\t\"Life saver\"\t10\tJune 3, 2017\t20\n"
)

@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "drugsComTrain_raw.tsv").write_text(TSV_CONTENT_A)
    (tmp_path / "drugsComTest_raw.tsv").write_text(TSV_CONTENT_B)
    return str(tmp_path)

def test_load_raw_merges_train_test(raw_dir):
    config = {"data": {"raw_dir": raw_dir,
                       "train_file": "drugsComTrain_raw.tsv",
                       "test_file": "drugsComTest_raw.tsv"}}
    df = load_raw(config)
    assert len(df) == 3
    assert set(df.columns) >= {"drugName", "condition", "review", "rating", "date", "usefulCount"}

def test_load_raw_resets_index(raw_dir):
    config = {"data": {"raw_dir": raw_dir,
                       "train_file": "drugsComTrain_raw.tsv",
                       "test_file": "drugsComTest_raw.tsv"}}
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

def test_filter_includes_drug_match():
    df = make_df([
        ["Buprenorphine", "Back Pain", "review text", 6, "Jan 1, 2015", 2],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1

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
