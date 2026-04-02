"""Evaluation: compare three classification approaches."""
import os
import sys
import pandas as pd
from sklearn.metrics import classification_report


LABELS = ["HIGH", "MODERATE", "LOW"]


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Return per-class precision/recall/F1 dict from classification_report."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return report  # type: ignore[return-value]


def compare_approaches(df: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline and cluster approaches against LLM labels (pseudo-ground-truth).

    Returns summary DataFrame with columns: approach, HIGH_f1, MODERATE_f1, LOW_f1, macro_f1.
    """
    rows = []
    y_true = df["llm_risk"].tolist()

    for approach_col, approach_name in [
        ("baseline_risk", "baseline_vs_llm"),
        ("risk_level", "cluster_vs_llm"),
    ]:
        y_pred = df[approach_col].tolist()
        report = compute_metrics(y_true, y_pred, LABELS)
        rows.append({
            "approach": approach_name,
            "HIGH_f1": report.get("HIGH", {}).get("f1-score", 0.0),
            "MODERATE_f1": report.get("MODERATE", {}).get("f1-score", 0.0),
            "LOW_f1": report.get("LOW", {}).get("f1-score", 0.0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        })

    return pd.DataFrame(rows)


def save_comparison(df_metrics: pd.DataFrame, config: dict) -> None:
    """Save approach comparison table to outputs/tables/approach_comparison.csv."""
    os.makedirs(config["paths"]["tables"], exist_ok=True)
    path = os.path.join(config["paths"]["tables"], "approach_comparison.csv")
    df_metrics.to_csv(path, index=False)
    print(f"Saved approach comparison to {path}", file=sys.stderr)


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)

    metrics_df = compare_approaches(df)
    print(metrics_df.to_string(), file=sys.stderr)
    save_comparison(metrics_df, config)
