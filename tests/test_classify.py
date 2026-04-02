import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.classify import classify_with_llm, batch_classify_llm

CONFIG = {
    "llm": {"model": "llama-3.1-8b-instant", "temperature": 0.1, "prompt_batch": 2, "classify_sample": 4},
    "random_seed": 42,
}

MOCK_RESULTS = [
    {"stage_label": "SEEKING_STABILITY", "risk_level": "MODERATE", "rationale": "Patient is newly on medication."},
    {"stage_label": "MAINTENANCE", "risk_level": "LOW", "rationale": "Patient is stable."},
]


def make_mock_response(results: list) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = json.dumps(results)
    return mock


def test_classify_with_llm_returns_required_keys():
    with patch("src.classify.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = make_mock_response(MOCK_RESULTS[:1])
        results = classify_with_llm(["i just started suboxone"], CONFIG)
    assert len(results) == 1
    assert "stage_label" in results[0]
    assert "risk_level" in results[0]
    assert "rationale" in results[0]


def test_classify_with_llm_handles_parse_error():
    mock = MagicMock()
    mock.choices[0].message.content = "not json"
    with patch("src.classify.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = mock
        results = classify_with_llm(["some review"], CONFIG)
    assert results[0]["risk_level"] == "MODERATE"


def test_classify_with_llm_batches_multiple():
    with patch("src.classify.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = make_mock_response(MOCK_RESULTS)
        results = classify_with_llm(["review one", "review two"], CONFIG)
    assert len(results) == 2
    assert results[0]["risk_level"] == "MODERATE"
    assert results[1]["risk_level"] == "LOW"


def test_batch_classify_llm_adds_columns():
    df = pd.DataFrame({"clean_review": ["review one", "review two", "review three", "review four"]})
    with patch("src.classify.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = make_mock_response(MOCK_RESULTS)
        result = batch_classify_llm(df, CONFIG)
    assert "llm_risk" in result.columns
    assert "llm_stage" in result.columns
    assert "llm_rationale" in result.columns
