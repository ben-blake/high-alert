"""LLM-based risk classification with stage labels and rationale."""
import json
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

CLASSIFY_PROMPT = """You are a public health data analyst. Classify each patient review of an addiction treatment medication below.

{reviews}

Return a JSON array of exactly {n} objects in the same order as the reviews:
[{{"stage_label": "STAGE", "risk_level": "HIGH|MODERATE|LOW", "rationale": "one sentence"}}, ...]
Output only the JSON array, nothing else."""


def classify_with_llm(texts: list, config: dict) -> list:
    """Classify a batch of review texts in a single API call.

    Returns list of dicts with stage_label, risk_level, rationale (one per text).
    """
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]

    reviews_text = "\n".join(f'[{i + 1}] "{t[:400]}"' for i, t in enumerate(texts))
    prompt = CLASSIFY_PROMPT.format(n=len(texts), reviews=reviews_text)

    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    try:
        parsed = json.loads(response.choices[0].message.content or "")
        if not isinstance(parsed, list) or len(parsed) != len(texts):
            raise ValueError("Wrong number of results")
        for item in parsed:
            if "risk_level" not in item:
                raise ValueError("Missing risk_level")
    except (json.JSONDecodeError, ValueError):
        parsed = [
            {"stage_label": "UNKNOWN", "risk_level": "MODERATE", "rationale": "Could not parse LLM response."}
            for _ in texts
        ]
    return parsed


def batch_classify_llm(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Sample reviews, LLM-classify them, add llm_risk/llm_stage/llm_rationale columns.

    Only classifies classify_sample rows (for evaluation); rest default to MODERATE.
    """
    df = df.copy()
    sample_n = config["llm"].get("classify_sample", len(df))
    prompt_batch = config["llm"].get("prompt_batch", 5)
    seed = config.get("random_seed", 42)

    df["llm_risk"] = "MODERATE"
    df["llm_stage"] = "UNKNOWN"
    df["llm_rationale"] = ""

    sample_df = df.sample(min(sample_n, len(df)), random_state=seed)
    texts = sample_df["clean_review"].tolist()
    indices = sample_df.index.tolist()

    all_results: list = []
    for i in tqdm(range(0, len(texts), prompt_batch), desc="LLM classification"):
        batch = texts[i : i + prompt_batch]
        all_results.extend(classify_with_llm(batch, config))

    for idx, result in zip(indices, all_results):
        df.at[idx, "llm_risk"] = result.get("risk_level", "MODERATE")
        df.at[idx, "llm_stage"] = result.get("stage_label", "UNKNOWN")
        df.at[idx, "llm_rationale"] = result.get("rationale", "")

    return df


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    sample_n = config["llm"].get("classify_sample", len(df))
    print(f"LLM-classifying {min(sample_n, len(df))} of {len(df)} reviews...", file=sys.stderr)
    df = batch_classify_llm(df, config)
    df.to_parquet(parquet_path, index=False)
    print(df["llm_risk"].value_counts().to_string(), file=sys.stderr)
    print("LLM classification saved.", file=sys.stderr)
