import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from unittest.mock import patch
from src.clustering import reduce_dimensions, cluster_hdbscan, cluster_kmeans

CONFIG = {
    "clustering": {
        "umap_n_components_cluster": 5,
        "umap_n_components_viz": 2,
        "umap_n_neighbors": 5,
        "umap_min_dist": 0.1,
        "hdbscan_min_cluster_size": 5,
        "hdbscan_min_samples": 2,
        "kmeans_random_seed": 42,
        "umap_random_seed": 42,
        "stage_sample_size": 3,
    },
    "random_seed": 42,
    "llm": {
        "model": "llama3.1:8b",
        "temperature": 0.1,
    },
}


def test_reduce_dimensions_output_shape():
    rng = np.random.default_rng(42)
    embeddings = rng.random((100, 50)).astype(np.float32)
    reduced = reduce_dimensions(embeddings, n_components=5, config=CONFIG)
    assert reduced.shape == (100, 5)


def test_hdbscan_returns_valid_labels():
    X, _ = make_blobs(n_samples=200, centers=4, n_features=5, random_state=42)  # type: ignore[misc]
    labels = cluster_hdbscan(X.astype(np.float32), CONFIG)
    assert len(labels) == 200
    assert all(l >= -1 for l in labels)


def test_kmeans_returns_n_clusters():
    X, _ = make_blobs(n_samples=150, centers=3, n_features=5, random_state=42)  # type: ignore[misc]
    labels = cluster_kmeans(X.astype(np.float32), n_clusters=3, config=CONFIG)
    assert len(labels) == 150
    assert len(set(labels)) == 3


# Task 9: LLM stage labeling tests

from src.clustering import label_clusters_with_llm, assign_cluster_labels

MOCK_LLM_RESPONSE = json.dumps({
    "stage_name": "SEEKING_STABILITY",
    "ttm_stage": "Action",
    "description": "Patients newly on medication, cautiously optimistic.",
    "risk_level": "MODERATE"
})


def test_label_clusters_with_llm_mock():
    rng = np.random.default_rng(0)
    n = 30
    embeddings = rng.random((n, 384)).astype(np.float32)
    df = pd.DataFrame({
        "clean_review": [f"review {i}" for i in range(n)],
        "drugName": ["Suboxone"] * n,
        "rating": [8] * n,
    })
    cluster_labels = np.array([0] * 15 + [1] * 15)

    from unittest.mock import MagicMock
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = MOCK_LLM_RESPONSE
    mock_stream = iter([mock_chunk])

    with patch("src.clustering.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = mock_stream
        result = label_clusters_with_llm(df, cluster_labels, embeddings, CONFIG)

    assert 0 in result
    assert 1 in result
    assert result[0]["stage_name"] == "SEEKING_STABILITY"
    assert result[0]["risk_level"] == "MODERATE"


def test_assign_cluster_labels():
    df = pd.DataFrame({"clean_review": ["r1", "r2", "r3"]})
    cluster_labels = np.array([0, 1, -1])
    cluster_stages = {
        0: {"stage_name": "RELAPSE", "ttm_stage": "Relapse", "description": "x", "risk_level": "HIGH"},
        1: {"stage_name": "MAINTENANCE", "ttm_stage": "Maintenance", "description": "y", "risk_level": "LOW"},
    }
    result = assign_cluster_labels(df, cluster_labels, cluster_stages)
    assert result.iloc[0]["stage_name"] == "RELAPSE"
    assert result.iloc[1]["stage_name"] == "MAINTENANCE"
    assert result.iloc[2]["stage_name"] == "NOISE"
    assert result.iloc[2]["risk_level"] == "MODERATE"
