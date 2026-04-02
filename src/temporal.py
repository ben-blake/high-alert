"""Temporal analysis: stage drift, drug trends, spike detection, narratives, plots."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def compute_stage_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % of each stage per year_quarter.

    Returns DataFrame: index=year_quarter (sorted), columns=stage_names, values=% share.
    """
    counts = df.groupby(["year_quarter", "stage_name"]).size().unstack(fill_value=0)
    dist = counts.div(counts.sum(axis=1), axis=0) * 100
    dist = dist.sort_index()
    return dist


def compute_drug_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median rating and dominant stage per drug per year_quarter."""
    agg = df.groupby(["drugName", "year_quarter"]).agg(
        median_rating=("rating", "median"),
        review_count=("clean_review", "count"),
        dominant_stage=("stage_name", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
    ).reset_index()
    return agg


def detect_spikes(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Detect quarters with anomalously high HIGH-risk review counts.

    Returns:
        time_series: DataFrame with year_quarter, high_count, z_score columns
        spike_quarters: list of year_quarter strings where z_score > threshold
    """
    window = config["temporal"]["spike_window"]
    threshold = config["temporal"]["spike_threshold"]

    high_df = df[df["baseline_risk"] == "HIGH"]
    counts = high_df.groupby("year_quarter").size().reset_index(name="high_count")
    counts = counts.sort_values("year_quarter").reset_index(drop=True)

    # Use global z-score for robust spike detection across the full time series.
    # Rolling stats can dilute spikes when the anomalous point falls inside the window.
    # spike_window is used for smoothing the series before scoring.
    smoothed = counts["high_count"].rolling(window=window, min_periods=1, center=False).mean()
    global_mean = smoothed.mean()
    global_std = smoothed.std()

    if global_std > 0:
        counts["z_score"] = (counts["high_count"] - global_mean) / global_std
    else:
        counts["z_score"] = 0.0

    spike_quarters = counts.loc[counts["z_score"] > threshold, "year_quarter"].tolist()
    return counts, spike_quarters


SPIKE_NARRATIVE_PROMPT = """You are a public health analyst reviewing addiction treatment trends.
The following {n} patient reviews were posted during {quarter}, a period when high-risk language spiked significantly.

Reviews (sorted by community usefulness):
{reviews}

Write exactly 3 sentences describing:
1. What the community was primarily discussing
2. What recovery challenges are evident
3. The public health significance of this spike"""


def generate_spike_narratives(
    df: pd.DataFrame, spike_quarters: list[str], config: dict
) -> dict[str, str]:
    """Generate LLM narratives for each spike quarter.

    For each spike, reads top spike_sample_size HIGH-risk reviews sorted by usefulCount descending.
    Returns dict mapping year_quarter -> narrative string.
    """
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]
    n_samples = config["llm"]["spike_sample_size"]
    narratives: dict[str, str] = {}

    for quarter in spike_quarters:
        mask = (df["year_quarter"] == quarter) & (df["baseline_risk"] == "HIGH")
        quarter_df = df.loc[mask].sort_values(by="usefulCount", ascending=False).head(n_samples)  # type: ignore[call-overload]

        reviews_text = "\n".join(
            f"- {row['clean_review'][:200]}" for _, row in quarter_df.iterrows()
        )
        prompt = SPIKE_NARRATIVE_PROMPT.format(
            n=len(quarter_df), quarter=quarter, reviews=reviews_text
        )
        print(f"Generating narrative for spike quarter {quarter}...", file=sys.stderr)
        client = Groq()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        narratives[quarter] = response.choices[0].message.content.strip()  # type: ignore[union-attr]

    return narratives


def plot_spike_detection(
    time_series: pd.DataFrame,
    spike_quarters: list[str],
    narratives: dict[str, str],
    config: dict,
) -> None:
    """Save spike detection hero chart to outputs/figures/spike_detection.png."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        range(len(time_series)),
        time_series["high_count"],
        color="#d62728", linewidth=2, marker="o", markersize=4, label="HIGH-risk reviews",
    )

    midpoint = len(time_series) / 2
    for quarter in spike_quarters:
        matches = time_series[time_series["year_quarter"] == quarter]
        if len(matches):
            idx = matches.index[0]
            ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.2, color="#d62728")
            narrative = narratives.get(quarter, "")
            if narrative:
                # Skip preamble lines; use first sentence that is not a heading
                sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if len(s.strip()) > 20]
                short = (sentences[0] + ".") if sentences else quarter
                on_right = idx > midpoint
                x_offset = -1.5 if on_right else 1.5
                ha = "right" if on_right else "left"
                ax.annotate(
                    f"↑ {quarter}\n{short[:55]}",
                    xy=(idx, time_series.loc[idx, "high_count"]),
                    xytext=(idx + x_offset, time_series.loc[idx, "high_count"] * 1.15),
                    fontsize=7, color="#d62728", ha=ha,
                    arrowprops=dict(arrowstyle="->", color="#d62728"),
                )

    tick_step = max(1, len(time_series) // 10)
    ax.set_xticks(range(0, len(time_series), tick_step))
    ax.set_xticklabels(
        time_series["year_quarter"].iloc[::tick_step], rotation=45, ha="right", fontsize=8
    )
    ax.set_xlabel("Quarter")
    ax.set_ylabel("HIGH-risk Review Count")
    ax.set_title("Temporal Spike Detection in Addiction Treatment Reviews", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    os.makedirs(config["paths"]["figures"], exist_ok=True)
    path = os.path.join(config["paths"]["figures"], "spike_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spike detection chart to {path}", file=sys.stderr)


def plot_stage_drift(stage_dist: pd.DataFrame, config: dict) -> None:
    """Save stacked area chart of stage distribution over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    stage_dist.plot.area(ax=ax, alpha=0.75, colormap="tab10")
    tick_step = max(1, len(stage_dist) // 10)
    ax.set_xticks(range(0, len(stage_dist), tick_step))
    ax.set_xticklabels(stage_dist.index[::tick_step], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("% of Reviews")
    ax.set_title("Recovery Stage Distribution Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    path = os.path.join(config["paths"]["figures"], "stage_drift.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved stage drift chart to {path}", file=sys.stderr)


def plot_drug_trends(drug_trends: pd.DataFrame, config: dict, top_n: int = 6) -> None:
    """Save drug effectiveness trends chart for top N drugs by review count."""
    top_drugs = (
        drug_trends.groupby("drugName")["review_count"].sum()
        .nlargest(top_n).index.tolist()
    )
    filtered = drug_trends[drug_trends["drugName"].isin(top_drugs)]
    all_quarters = sorted(drug_trends["year_quarter"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    for drug in top_drugs:
        drug_df = filtered[filtered["drugName"] == drug].sort_values("year_quarter")
        ax.plot(drug_df["year_quarter"], drug_df["median_rating"], label=drug, linewidth=1.5, marker=".")

    tick_step = max(1, len(all_quarters) // 10)
    ax.set_xticks(range(0, len(all_quarters), tick_step))
    ax.set_xticklabels(
        [all_quarters[i] for i in range(0, len(all_quarters), tick_step)],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Median Patient Rating (1–10)")
    ax.set_title("Treatment Drug Effectiveness Trends Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(config["paths"]["figures"], "drug_trends.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved drug trends chart to {path}", file=sys.stderr)


def plot_umap_clusters(df: pd.DataFrame, cluster_stages: dict, config: dict) -> None:
    """Save 2D UMAP scatter plot colored by recovery stage."""
    fig, ax = plt.subplots(figsize=(12, 8))
    stages = df["stage_name"].unique()
    palette = sns.color_palette("tab10", len(stages))
    color_map = dict(zip(stages, palette))

    for stage in stages:
        mask = df["stage_name"] == stage
        ax.scatter(
            df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
            c=[color_map[stage]], label=stage, s=5, alpha=0.6,
        )

    ax.set_title("Recovery Stage Clusters (UMAP 2D Projection)", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(fontsize=8, markerscale=3)
    plt.tight_layout()

    path = os.path.join(config["paths"]["figures"], "umap_clusters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved UMAP cluster chart to {path}", file=sys.stderr)


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)

    with open(os.path.join(config["paths"]["tables"], "cluster_stages.json")) as f:
        cluster_stages = json.load(f)

    print("Computing stage distribution...", file=sys.stderr)
    stage_dist = compute_stage_distribution(df)
    os.makedirs(config["paths"]["figures"], exist_ok=True)
    plot_stage_drift(stage_dist, config)

    print("Computing drug trends...", file=sys.stderr)
    drug_trends = compute_drug_trends(df)
    os.makedirs(config["paths"]["tables"], exist_ok=True)
    drug_trends.to_csv(os.path.join(config["paths"]["tables"], "drug_trends.csv"), index=False)
    plot_drug_trends(drug_trends, config)

    print("Detecting spikes...", file=sys.stderr)
    time_series, spike_quarters = detect_spikes(df, config)
    print(f"Spike quarters: {spike_quarters}", file=sys.stderr)

    narratives = generate_spike_narratives(df, spike_quarters, config)
    os.makedirs(config["paths"]["summaries"], exist_ok=True)
    with open(os.path.join(config["paths"]["summaries"], "spike_narratives.json"), "w") as f:
        json.dump(narratives, f, indent=2)

    plot_spike_detection(time_series, spike_quarters, narratives, config)
    plot_umap_clusters(df, cluster_stages, config)

    print("Temporal analysis complete.", file=sys.stderr)
