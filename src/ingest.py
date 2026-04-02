"""Data ingestion: loading, filtering, temporal feature extraction."""
import os
import sys
import pandas as pd
import yaml


def load_raw(config: dict) -> pd.DataFrame:
    """Load and merge train/test CSV files into a single DataFrame."""
    raw_dir = config["data"]["raw_dir"]
    train_path = os.path.join(raw_dir, config["data"]["train_file"])
    test_path = os.path.join(raw_dir, config["data"]["test_file"])
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    df = pd.concat([train, test], ignore_index=True)
    return df


def filter_addiction_related(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep rows matching addiction conditions (substring) OR addiction drugs (substring, case-insensitive)."""
    conditions = [c.lower() for c in config["filter"]["addiction_conditions"]]
    drugs = [d.lower() for d in config["filter"]["addiction_drugs"]]

    cond_col = df["condition"].fillna("").str.lower()
    drug_col = df["drugName"].fillna("").str.lower()

    condition_mask = cond_col.apply(lambda x: any(c in x for c in conditions))
    drug_mask = drug_col.apply(lambda x: any(d in x for d in drugs))

    return pd.DataFrame(df[condition_mask | drug_mask]).reset_index(drop=True)


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date column and add year, quarter, year_quarter columns."""
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df["year"] = df["date_parsed"].dt.year
    df["quarter"] = df["date_parsed"].dt.quarter
    df["year_quarter"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Print exploratory statistics to stderr."""
    print(f"\n=== EDA ===", file=sys.stderr)
    print(f"Total reviews: {len(df)}", file=sys.stderr)
    min_date = df['date_parsed'].min()
    max_date = df['date_parsed'].max()
    date_range = f"{min_date.date()} to {max_date.date()}" if pd.notna(min_date) and pd.notna(max_date) else "unknown"
    print(f"Date range: {date_range}", file=sys.stderr)
    print(f"Unique drugs: {df['drugName'].nunique()}", file=sys.stderr)
    print(f"Unique conditions: {df['condition'].nunique()}", file=sys.stderr)
    valid_yq = df['year_quarter'].dropna().loc[~df['year_quarter'].str.startswith('nan', na=True)]
    yq_range = f"{valid_yq.min()} to {valid_yq.max()}" if len(valid_yq) > 0 else "unknown"
    print(f"Year-quarter range: {yq_range}", file=sys.stderr)
    print(f"\nTop 10 drugs by review count:", file=sys.stderr)
    print(df["drugName"].value_counts().head(10).to_string(), file=sys.stderr)
    print(f"\nTop conditions:", file=sys.stderr)
    print(df["condition"].value_counts().head(10).to_string(), file=sys.stderr)
    print(f"\nRating distribution:\n{df['rating'].describe()}", file=sys.stderr)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print("Loading raw data...", file=sys.stderr)
    df = load_raw(config)
    print(f"Raw rows: {len(df)}", file=sys.stderr)

    df = filter_addiction_related(df, config)
    print(f"After filter: {len(df)} rows", file=sys.stderr)

    df = extract_temporal_features(df)
    run_eda(df)

    out_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}", file=sys.stderr)
