"""LLM-based risk classification with stage labels and rationale."""
import json
import sys
from typing import Any, Mapping, cast
import pandas as pd
import ollama
from tqdm import tqdm

CLASSIFY_PROMPT = """You are a public health data analyst. Read this patient review of an addiction treatment medication.
Classify the review into a recovery stage and risk level.

Review: "{review}"

Respond with ONLY valid JSON:
{{
  "stage_label": "SHORT_STAGE_NAME (e.g. SEEKING_STABILITY, RELAPSE_RISK, IN_RECOVERY, EARLY_TREATMENT, MAINTENANCE)",
  "risk_level": "HIGH, MODERATE, or LOW",
  "rationale": "One sentence explaining the classification."
}}"""


def classify_with_llm(texts: list, config: dict) -> list:
    """Classify a list of review texts. Returns list of dicts with stage_label, risk_level, rationale."""
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]
    results = []

    for text in texts:
        prompt = CLASSIFY_PROMPT.format(review=text[:500])
        raw = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        response = cast(Mapping[str, Any], raw)
        try:
            parsed = json.loads(response["message"]["content"])
            if "risk_level" not in parsed:
                raise ValueError("Missing risk_level")
        except (json.JSONDecodeError, ValueError):
            parsed = {
                "stage_label": "UNKNOWN",
                "risk_level": "MODERATE",
                "rationale": "Could not parse LLM response.",
            }
        results.append(parsed)

    return results


def batch_classify_llm(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add llm_risk, llm_stage, llm_rationale columns. Processes in batches."""
    df = df.copy()
    batch_size = config["llm"]["batch_size"]
    all_results = []

    texts = df["clean_review"].tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc="LLM classification"):
        batch = texts[i : i + batch_size]
        all_results.extend(classify_with_llm(batch, config))

    df["llm_risk"] = [r["risk_level"] for r in all_results]
    df["llm_stage"] = [r.get("stage_label", "UNKNOWN") for r in all_results]
    df["llm_rationale"] = [r.get("rationale", "") for r in all_results]
    return df


if __name__ == "__main__":
    import os
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    print(f"Classifying {len(df)} reviews with LLM...", file=sys.stderr)
    df = batch_classify_llm(df, config)
    df.to_parquet(parquet_path, index=False)
    print(df["llm_risk"].value_counts().to_string(), file=sys.stderr)
    print("LLM classification saved.", file=sys.stderr)
