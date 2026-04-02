"""Dimensionality reduction and clustering for recovery stage discovery."""
import sys
import json
import numpy as np
import pandas as pd
import umap
import hdbscan
import ollama
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


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


STAGE_LABEL_PROMPT = """You are analyzing patient reviews of addiction treatment medications.
Below are {n} representative reviews from one behavioral cluster.

Reviews:
{reviews}

Based on these reviews, identify the recovery stage this cluster represents.
Respond with ONLY valid JSON in this exact format:
{{
  "stage_name": "SHORT_STAGE_NAME_IN_CAPS",
  "ttm_stage": "One of: Pre-Contemplation, Contemplation, Preparation, Action, Maintenance, Relapse",
  "description": "One sentence describing this group of patients.",
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
        sample_reviews = df.iloc[top_idx]["clean_review"].tolist()

        prompt = STAGE_LABEL_PROMPT.format(
            n=len(sample_reviews),
            reviews="\n".join(f"- {r}" for r in sample_reviews),
        )
        print(f"Labeling cluster {cluster_id}...", file=sys.stderr)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": config["llm"]["temperature"]},
        )
        try:
            parsed = json.loads(response["message"]["content"])  # type: ignore[index]
        except json.JSONDecodeError:
            parsed = {
                "stage_name": f"CLUSTER_{cluster_id}",
                "ttm_stage": "Unknown",
                "description": "Could not parse LLM response.",
                "risk_level": "MODERATE",
            }
        results[int(cluster_id)] = parsed

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
