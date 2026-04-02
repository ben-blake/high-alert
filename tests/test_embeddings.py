import numpy as np
import pandas as pd
import pytest
import os
from unittest.mock import patch, MagicMock
from src.embeddings import generate_embeddings, load_or_generate_embeddings

def test_generate_embeddings_shape():
    texts = ["I feel great", "relapsed again", "sober for a year", "hopeless", "in recovery"]
    embeddings = generate_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=2)
    assert embeddings.shape == (5, 384)
    assert embeddings.dtype == np.float32

def test_generate_embeddings_deterministic():
    texts = ["consistent output test"]
    e1 = generate_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=1)
    e2 = generate_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=1)
    np.testing.assert_array_almost_equal(e1, e2)

def test_cache_hit_skips_generation(tmp_path):
    """Second call loads from disk without calling SentenceTransformer."""
    df = pd.DataFrame({"clean_review": ["text one", "text two", "text three"]})
    config = {
        "data": {"processed_dir": str(tmp_path),
                 "embeddings_file": "embeddings.npy",
                 "embedding_index_file": "embedding_index.json"},
        "embeddings": {"model": "all-MiniLM-L6-v2", "batch_size": 64},
    }
    # First call generates
    emb1, idx1 = load_or_generate_embeddings(df, config)

    # Second call should load from cache — patch SentenceTransformer to assert it's not called
    with patch("src.embeddings.SentenceTransformer") as mock_st:
        emb2, idx2 = load_or_generate_embeddings(df, config)
        mock_st.assert_not_called()

    np.testing.assert_array_equal(emb1, emb2)
    assert idx1 == idx2
