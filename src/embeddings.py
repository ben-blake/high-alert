"""Embedding generation with disk caching."""
import os
import sys
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_embeddings(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    """Generate sentence embeddings. Returns float32 array of shape (N, dim)."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings).astype(np.float32)


def load_or_generate_embeddings(
    df: pd.DataFrame, config: dict
) -> tuple[np.ndarray, list[int]]:
    """Load embeddings from cache if valid, otherwise generate and cache.

    Returns:
        embeddings: float32 array (N, dim)
        row_indices: list of DataFrame indices corresponding to embedding rows
    """
    processed_dir = config["data"]["processed_dir"]
    emb_path = os.path.join(processed_dir, config["data"]["embeddings_file"])
    idx_path = os.path.join(processed_dir, config["data"]["embedding_index_file"])

    if os.path.exists(emb_path) and os.path.exists(idx_path):
        with open(idx_path) as f:
            meta = json.load(f)
        if meta.get("n_rows") == len(df):
            print("Loading embeddings from cache...", file=sys.stderr)
            embeddings = np.load(emb_path)
            return embeddings, meta["row_indices"]
        print("Cache row count mismatch — regenerating.", file=sys.stderr)

    texts = df["clean_review"].tolist()
    row_indices = list(df.index)
    print(f"Generating embeddings for {len(texts)} reviews...", file=sys.stderr)
    embeddings = generate_embeddings(
        texts,
        model_name=config["embeddings"]["model"],
        batch_size=config["embeddings"]["batch_size"],
    )
    os.makedirs(processed_dir, exist_ok=True)
    np.save(emb_path, embeddings)
    with open(idx_path, "w") as f:
        json.dump({"n_rows": len(df), "row_indices": row_indices}, f)
    print(f"Embeddings saved to {emb_path}", file=sys.stderr)
    return embeddings, row_indices


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    embeddings, _ = load_or_generate_embeddings(df, config)
    print(f"Embeddings shape: {embeddings.shape}", file=sys.stderr)
