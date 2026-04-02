"""Keyword/regex baseline risk classifier."""
import re
import sys
import pandas as pd


def classify_risk_baseline(text: str, config: dict) -> str:
    """Classify a single review as HIGH, LOW, or MODERATE using keyword rules.

    HIGH takes priority over LOW.
    """
    text_lower = text.lower()
    for pattern in config["baseline"]["high_risk_patterns"]:
        if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
            return "HIGH"
    for pattern in config["baseline"]["low_risk_patterns"]:
        if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
            return "LOW"
    return "MODERATE"


def batch_classify_baseline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add baseline_risk column to df."""
    df = df.copy()
    df["baseline_risk"] = df["clean_review"].apply(lambda t: classify_risk_baseline(t, config))
    return df


if __name__ == "__main__":
    import os
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    df = batch_classify_baseline(df, config)
    print(df["baseline_risk"].value_counts().to_string(), file=sys.stderr)
    df.to_parquet(parquet_path, index=False)
    print("Baseline classification saved.", file=sys.stderr)
