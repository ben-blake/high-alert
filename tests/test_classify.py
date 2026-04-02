import pandas as pd
import json
from unittest.mock import patch
from src.classify import classify_with_llm, batch_classify_llm

CONFIG = {
    "llm": {"model": "llama3.1:8b", "batch_size": 2, "temperature": 0.1}
}

MOCK_RESPONSE_JSON = json.dumps({
    "stage_label": "SEEKING_STABILITY",
    "risk_level": "MODERATE",
    "rationale": "Patient is newly on medication and cautiously optimistic."
})

def make_mock_response(content: str):
    return {"message": {"content": content}}

def test_classify_with_llm_returns_required_keys():
    with patch("src.classify.ollama.chat", return_value=make_mock_response(MOCK_RESPONSE_JSON)):
        results = classify_with_llm(["i just started suboxone"], CONFIG)
    assert len(results) == 1
    assert "stage_label" in results[0]
    assert "risk_level" in results[0]
    assert "rationale" in results[0]

def test_classify_with_llm_handles_parse_error():
    with patch("src.classify.ollama.chat", return_value=make_mock_response("not json")):
        results = classify_with_llm(["some review"], CONFIG)
    assert results[0]["risk_level"] == "MODERATE"

def test_batch_classify_llm_adds_columns():
    df = pd.DataFrame({"clean_review": ["review one", "review two"]})
    with patch("src.classify.ollama.chat", return_value=make_mock_response(MOCK_RESPONSE_JSON)):
        result = batch_classify_llm(df, CONFIG)
    assert "llm_risk" in result.columns
    assert "llm_stage" in result.columns
    assert "llm_rationale" in result.columns
