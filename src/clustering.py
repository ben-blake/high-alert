"""Dimensionality reduction and clustering for recovery stage discovery."""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import umap
import hdbscan
from dotenv import load_dotenv
from groq import Groq
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def reduce_dimensions(embeddings: np.ndarray, n_components: int, config: dict) -> np.ndarray:
    """Reduce embedding dimensions with UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=config["clustering"]["umap_n_neighbors"],
        min_dist=config["clustering"]["umap_min_dist"],
        random_state=config["clustering"]["umap_random_seed"],
        metric="cosine",
    )
    return np.asarray(reducer.fit_transform(embeddings)).astype(np.float32)


def cluster_hdbscan(embeddings_reduced: np.ndarray, config: dict) -> np.ndarray:
    """Cluster with HDBSCAN. Returns label array; -1 = noise."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config["clustering"]["hdbscan_min_cluster_size"],
        min_samples=config["clustering"]["hdbscan_min_samples"],
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings_reduced)


def cluster_kmeans(embeddings_reduced: np.ndarray, n_clusters: int, config: dict) -> np.ndarray:
    """Cluster with KMeans. Returns label array."""
    km = KMeans(
        n_clusters=n_clusters,
        random_state=config["clustering"]["kmeans_random_seed"],
        n_init="auto",
    )
    return km.fit_predict(embeddings_reduced)


STAGE_LABEL_PROMPT = """You are a public health analyst classifying addiction treatment patient reviews.
Below are {n} reviews (with patient ratings 1-10) from one group discovered through semantic clustering.

Reviews:
{reviews}

Classify this group using the Transtheoretical Model (TTM). Use the full spectrum:
- Pre-Contemplation: not yet considering change, in denial, or unaware of need
- Contemplation: aware of problem, thinking about change, ambivalent
- Preparation: committed to change, early medication use, may still have slip-ups
- Action: actively working to change, within first 6 months, effortful
- Maintenance: sustained change 6+ months, stable, confident
- Relapse: returned to prior behavior after a period of change, discouraged

Risk level criteria:
- HIGH: mentions relapse, overdose, giving up, hopeless, still using, can't stop
- MODERATE: active struggle, uncertain, side effects threatening adherence, early recovery
- LOW: stable, long-term success, confident in sobriety

Note: if reviews are mixed (some hopeful, some struggling), reflect that in the risk level.

CRITICAL — stage_name rules:
- Must be UNIQUE and SPECIFIC to this cluster's dominant substance + behavior pattern
- Format: SUBSTANCE_BEHAVIOR (e.g., SMOKING_LONG_TERM_SUCCESS, SUBOXONE_EARLY_TAPER, NALTREXONE_STABLE, ALCOHOL_WITHDRAWAL_CRISIS, OPIOID_MAINTENANCE_STABLE)
- NEVER use a bare TTM stage name alone (MAINTENANCE, ACTION, etc.) — always qualify with substance or pattern
- If the group spans multiple substances, pick the most dominant one

Respond with ONLY valid JSON, no other text:
{{
  "stage_name": "SUBSTANCE_BEHAVIOR_IN_CAPS",
  "ttm_stage": "One of: Pre-Contemplation, Contemplation, Preparation, Action, Maintenance, Relapse",
  "description": "One sentence: what specific behavioral pattern defines this group?",
  "risk_level": "One of: HIGH, MODERATE, LOW"
}}"""


def label_clusters_with_llm(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
    config: dict,
) -> dict[int, dict]:
    """Label each cluster by having LLM read representative samples.

    Samples the stage_sample_size reviews closest to each cluster centroid
    (cosine similarity in 384-dim embedding space).
    Returns dict mapping cluster_id -> {stage_name, ttm_stage, description, risk_level}.
    """
    unique_clusters = [c for c in sorted(set(cluster_labels)) if c != -1]
    n_samples = config["clustering"]["stage_sample_size"]
    model = config["llm"]["model"]
    results: dict[int, dict] = {}

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_indices = np.where(mask)[0]

        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        top_n = min(n_samples, len(cluster_indices))
        top_idx = cluster_indices[np.argsort(sims)[-top_n:][::-1]]
        sample_rows = df.iloc[top_idx]
        reviews_text = "\n".join(
            f"- (rating: {row['rating']}/10) {row['clean_review'][:300]}"
            for _, row in sample_rows.iterrows()
        )

        prompt = STAGE_LABEL_PROMPT.format(
            n=len(sample_rows),
            reviews=reviews_text,
        )
        print(f"Labeling cluster {cluster_id}... ", end="", flush=True, file=sys.stderr)
        client = Groq()
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config["llm"]["temperature"],
            stream=True,
        )
        content = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            content += token
            print(token, end="", flush=True, file=sys.stderr)
        print(file=sys.stderr)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {
                "stage_name": f"CLUSTER_{cluster_id}",
                "ttm_stage": "Unknown",
                "description": "Could not parse LLM response.",
                "risk_level": "MODERATE",
            }
        results[int(cluster_id)] = parsed
        delay = config["llm"].get("api_delay", 0)
        if delay and cluster_id != unique_clusters[-1]:
            time.sleep(delay)

    return results


def assign_cluster_labels(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_stages: dict[int, dict],
) -> pd.DataFrame:
    """Add stage_name, ttm_stage, risk_level, cluster_id columns to df."""
    df = df.copy()
    df["cluster_id"] = cluster_labels

    def map_row(row):
        cid = int(row["cluster_id"])
        if cid == -1 or cid not in cluster_stages:
            return pd.Series({"stage_name": "NOISE", "ttm_stage": "Unknown", "risk_level": "MODERATE"})
        stage = cluster_stages[cid]
        return pd.Series({
            "stage_name": stage["stage_name"],
            "ttm_stage": stage["ttm_stage"],
            "risk_level": stage["risk_level"],
        })

    df[["stage_name", "ttm_stage", "risk_level"]] = df.apply(map_row, axis=1)
    return df


if __name__ == "__main__":
    import os
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    emb_path = os.path.join(config["data"]["processed_dir"], config["data"]["embeddings_file"])

    df = pd.read_parquet(parquet_path)
    embeddings = np.load(emb_path)

    print("Reducing dimensions (5-dim for clustering)...", file=sys.stderr)
    reduced_5d = reduce_dimensions(
        embeddings,
        n_components=config["clustering"]["umap_n_components_cluster"],
        config=config,
    )

    print("Reducing dimensions (2-dim for visualization)...", file=sys.stderr)
    reduced_2d = reduce_dimensions(
        embeddings,
        n_components=config["clustering"]["umap_n_components_viz"],
        config=config,
    )

    print("Running HDBSCAN...", file=sys.stderr)
    hdbscan_labels = cluster_hdbscan(reduced_5d, config)
    n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    print(
        f"HDBSCAN found {n_clusters} clusters, {(hdbscan_labels == -1).sum()} noise points",
        file=sys.stderr,
    )

    print("Running KMeans...", file=sys.stderr)
    kmeans_labels = cluster_kmeans(reduced_5d, n_clusters=n_clusters, config=config)

    print("Labeling clusters with LLM...", file=sys.stderr)
    cluster_stages = label_clusters_with_llm(df, hdbscan_labels, embeddings, config)

    os.makedirs(config["paths"]["tables"], exist_ok=True)
    with open(os.path.join(config["paths"]["tables"], "cluster_stages.json"), "w") as f:
        json.dump(cluster_stages, f, indent=2)
    print("Stage labels saved.", file=sys.stderr)

    df = assign_cluster_labels(df, hdbscan_labels, cluster_stages)
    df["umap_x"] = reduced_2d[:, 0]
    df["umap_y"] = reduced_2d[:, 1]
    df["kmeans_label"] = kmeans_labels
    df.to_parquet(parquet_path, index=False)
    print("Clustering complete. Parquet updated.", file=sys.stderr)
