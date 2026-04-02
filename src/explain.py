"""LLM-generated cluster summaries for public health analyst audience."""
import os
import sys
import json
import pandas as pd
import ollama

SUMMARY_PROMPT = """You are a public health data analyst preparing a briefing for a state health department.
Below is information about one behavioral cluster identified from {total} patient reviews of addiction treatment medications.

Stage Name: {stage_name}
TTM Stage Mapping: {ttm_stage}
Percentage of corpus: {pct:.1f}%
Number of reviews: {count}
Risk level: {risk_level}

Representative patient reviews:
{reviews}

Write a structured 4-sentence summary covering:
1. Who these patients are and where they are in their recovery journey
2. The primary treatment challenges or successes evident in this group
3. The public health significance of this population segment
4. One recommended action for health system planners based on this data"""


def generate_cluster_summaries(
    df: pd.DataFrame, cluster_stages: dict, config: dict
) -> str:
    """Generate markdown summaries for each cluster stage.

    Returns a markdown string with one section per cluster.
    """
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]
    total = len(df)
    sections = ["# Recovery Stage Cluster Summaries\n"]

    for cluster_id_str, stage_info in cluster_stages.items():
        stage_name = stage_info["stage_name"]
        cluster_df = df[df["stage_name"] == stage_name]
        count = len(cluster_df)
        pct = (count / total) * 100

        sample = cluster_df.sort_values(by="usefulCount", ascending=False).head(5)  # type: ignore[call-overload]
        reviews_text = "\n".join(f"- {row['clean_review'][:200]}" for _, row in sample.iterrows())

        prompt = SUMMARY_PROMPT.format(
            total=total,
            stage_name=stage_name,
            ttm_stage=stage_info.get("ttm_stage", "Unknown"),
            pct=pct,
            count=count,
            risk_level=stage_info.get("risk_level", "MODERATE"),
            reviews=reviews_text,
        )
        print(f"Generating summary for stage {stage_name}...", file=sys.stderr)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        sections.append(f"## {stage_name} ({stage_info.get('ttm_stage', '')})")
        sections.append(f"**Reviews:** {count} ({pct:.1f}% of corpus) | **Risk:** {stage_info.get('risk_level', 'MODERATE')}\n")
        sections.append(response["message"]["content"].strip())  # type: ignore[index]
        sections.append("")

    return "\n\n".join(sections)


def save_cluster_summaries(summaries: str, config: dict) -> None:
    """Save summaries markdown to outputs/summaries/cluster_summaries.md."""
    os.makedirs(config["paths"]["summaries"], exist_ok=True)
    path = os.path.join(config["paths"]["summaries"], "cluster_summaries.md")
    with open(path, "w") as f:
        f.write(summaries)
    print(f"Cluster summaries saved to {path}", file=sys.stderr)


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)

    with open(os.path.join(config["paths"]["tables"], "cluster_stages.json")) as f:
        cluster_stages = json.load(f)

    summaries = generate_cluster_summaries(df, cluster_stages, config)
    save_cluster_summaries(summaries, config)
