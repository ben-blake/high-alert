import pandas as pd
import pytest
from src.preprocess import clean_text, preprocess_reviews

def test_clean_text_html_apostrophe():
    assert clean_text("It&#039;s great") == "it's great"

def test_clean_text_html_amp():
    assert clean_text("pain &amp; anxiety") == "pain & anxiety"

def test_clean_text_html_quote():
    assert clean_text("&quot;miracle drug&quot;") == '"miracle drug"'

def test_clean_text_extra_whitespace():
    assert clean_text("  too   many   spaces  ") == "too many spaces"

def test_clean_text_lowercases():
    assert clean_text("GREAT Drug") == "great drug"

def test_clean_text_strips_span_tags():
    assert clean_text("condition</span>") == "condition"

def test_preprocess_reviews_adds_column():
    df = pd.DataFrame({"review": ["It&#039;s great", "Pain &amp; anxiety"]})
    result = preprocess_reviews(df)
    assert "clean_review" in result.columns
    assert result.iloc[0]["clean_review"] == "it's great"
    assert result.iloc[1]["clean_review"] == "pain & anxiety"

def test_preprocess_reviews_preserves_original():
    df = pd.DataFrame({"review": ["It&#039;s great"]})
    result = preprocess_reviews(df)
    assert "review" in result.columns  # original preserved
