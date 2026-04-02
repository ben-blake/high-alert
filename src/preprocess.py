"""Text preprocessing for drug reviews."""
import re
import sys
import html
import pandas as pd


def clean_text(text: str) -> str:
    """Decode HTML entities, strip tags, normalize whitespace, lowercase."""
    text = html.unescape(text)
    text = re.sub(r"</?\w+>", "", text)   # strip HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Add clean_review column with preprocessed text."""
    df = df.copy()
    df["clean_review"] = df["review"].fillna("").apply(clean_text)
    return df


if __name__ == "__main__":
    import os
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows", file=sys.stderr)

    df = preprocess_reviews(df)
    df.to_parquet(parquet_path, index=False)
    print(f"Preprocessing complete. Saved clean_review column to {parquet_path}", file=sys.stderr)
