# Recovery Trajectory Staging System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that discovers recovery stages from addiction treatment drug reviews, compares three classification approaches, and surfaces temporal risk signals for public health analysts.

**Architecture:** Kaggle KUC drug reviews → filter to addiction-related → embed with sentence-transformers → HDBSCAN discovers clusters → LLM labels each cluster as a recovery stage → three-approach comparison (keyword baseline / embedding / direct LLM) → temporal spike detection → Streamlit dashboard.

**Tech Stack:** Python 3.11+, pandas, sentence-transformers (all-MiniLM-L6-v2), umap-learn, hdbscan, scikit-learn, ollama (llama3.1:8b), matplotlib/seaborn, streamlit, pytest, pyyaml

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.yaml` | All hyperparameters, paths, filter lists, seeds |
| `requirements.txt` | Pinned dependencies |
| `.env.example` | API key template (Groq fallback) |
| `reproduce.sh` | Single-command pipeline run |
| `src/__init__.py` | Empty package marker |
| `src/ingest.py` | `load_raw()`, `filter_addiction_related()`, `extract_temporal_features()`, `run_eda()` |
| `src/preprocess.py` | `clean_text()`, `preprocess_reviews()` |
| `src/embeddings.py` | `generate_embeddings()`, `load_or_generate_embeddings()` |
| `src/clustering.py` | `reduce_dimensions()`, `cluster_hdbscan()`, `cluster_kmeans()`, `label_clusters_with_llm()`, `assign_cluster_labels()` |
| `src/baseline.py` | `classify_risk_baseline()`, `batch_classify_baseline()` |
| `src/classify.py` | `classify_with_llm()`, `batch_classify_llm()` |
| `src/evaluate.py` | `compute_metrics()`, `compare_approaches()`, `save_comparison()` |
| `src/temporal.py` | `compute_stage_distribution()`, `compute_drug_trends()`, `detect_spikes()`, `generate_spike_narratives()`, `plot_*()` |
| `src/explain.py` | `generate_cluster_summaries()`, `save_cluster_summaries()` |
| `app.py` | Streamlit dashboard (4 tabs) |
| `tests/test_ingest.py` | Unit tests for ingest module |
| `tests/test_preprocess.py` | Unit tests for preprocess module |
| `tests/test_embeddings.py` | Unit tests for embeddings module |
| `tests/test_clustering.py` | Unit tests for clustering module |
| `tests/test_baseline.py` | Unit tests for baseline classifier |
| `tests/test_classify.py` | Unit tests for LLM classifier |
| `tests/test_evaluate.py` | Unit tests for evaluation metrics |
| `tests/test_temporal.py` | Unit tests for temporal analysis |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/README.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p data/raw data/processed outputs/figures outputs/tables outputs/summaries src tests notebooks report video
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `requirements.txt`**

```
pandas==2.2.2
numpy==1.26.4
pyarrow==15.0.2
scikit-learn==1.4.2
sentence-transformers==2.7.0
umap-learn==0.5.6
hdbscan==0.8.38
matplotlib==3.8.4
seaborn==0.13.2
streamlit==1.33.0
ollama==0.2.1
pyyaml==6.0.1
python-dotenv==1.0.1
pytest==8.1.1
tqdm==4.66.2
```

- [ ] **Step 3: Create `config.yaml`**

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  train_file: drugsComTrain_raw.tsv
  test_file: drugsComTest_raw.tsv
  filtered_file: reviews_filtered.parquet
  embeddings_file: embeddings.npy
  embedding_index_file: embedding_index.json

filter:
  addiction_conditions:
    - "opiate dependence"
    - "opioid dependence"
    - "alcohol use disorder"
    - "benzodiazepine withdrawal"
    - "cocaine dependence"
    - "substance abuse"
    - "drug dependence"
    - "heroin dependence"
    - "opioid withdrawal"
    - "alcohol withdrawal"
    - "nicotine dependence"
    - "methamphetamine dependence"
  addiction_drugs:
    - "buprenorphine"
    - "methadone"
    - "naltrexone"
    - "suboxone"
    - "vivitrol"
    - "disulfiram"
    - "acamprosate"
    - "naloxone"
    - "subutex"
    - "zubsolv"
    - "varenicline"
    - "chantix"
    - "campral"

embeddings:
  model: all-MiniLM-L6-v2
  batch_size: 64

clustering:
  umap_n_components_cluster: 5
  umap_n_components_viz: 2
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  hdbscan_min_cluster_size: 30
  hdbscan_min_samples: 5
  kmeans_random_seed: 42
  umap_random_seed: 42
  stage_sample_size: 10

llm:
  model: llama3.1:8b
  batch_size: 10
  temperature: 0.1
  spike_sample_size: 20

baseline:
  high_risk_patterns:
    - "relapsed"
    - "relapse"
    - "overdosed"
    - "overdose"
    - "gave up"
    - "stopped working"
    - "addicted again"
    - "using again"
    - "back to using"
    - "failed"
    - "hopeless"
    - "suicidal"
    - "want to die"
  low_risk_patterns:
    - "sober"
    - "sobriety"
    - "clean for"
    - "changed my life"
    - "life-changing"
    - "highly recommend"
    - "in recovery"
    - "miracle"
    - "saved my life"

temporal:
  spike_window: 4
  spike_threshold: 2.0

paths:
  figures: outputs/figures
  tables: outputs/tables
  summaries: outputs/summaries

random_seed: 42
```

- [ ] **Step 4: Create `.env.example`**

```
# Optional: Groq API key for faster LLM inference fallback
# Get a free key at https://console.groq.com
GROQ_API_KEY=your_key_here
```

- [ ] **Step 5: Create `.gitignore`**

```
data/raw/
data/processed/
outputs/
.env
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
```

- [ ] **Step 6: Create `data/README.md`**

```markdown
# Dataset Information

## KUC Drug Review Dataset (Kaggle)

Source: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

### Download Instructions

1. Install kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   kaggle datasets download jessicali9530/kuc-hackathon-winter-2018 -p data/raw --unzip
   ```

### Files Expected in `data/raw/`
- `drugsComTrain_raw.tsv` — training split (~161k rows)
- `drugsComTest_raw.tsv` — test split (~53k rows)

### Columns
- `drugName` — name of the drug reviewed
- `condition` — medical condition being treated
- `review` — patient-written text review
- `rating` — patient satisfaction rating (1–10)
- `date` — date of review (format: "Month DD, YYYY")
- `usefulCount` — number of users who found the review helpful
```

- [ ] **Step 7: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 8: Commit**

```bash
git init
git add requirements.txt config.yaml .env.example .gitignore src/__init__.py tests/__init__.py data/README.md docs/
git commit -m "chore: project scaffolding, config, and directory structure"
```

---

## Task 2: Data Loading (`src/ingest.py`)

**Files:**
- Create: `src/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
import pandas as pd
import pytest
import os
import tempfile
from src.ingest import load_raw

TSV_CONTENT_A = (
    "\tdrugName\tcondition\treview\trating\tdate\tusefulCount\n"
    "0\tDrug A\tOpiate Dependence\t\"Great drug\"\t9\tJanuary 1, 2015\t10\n"
    "1\tDrug B\tDepression\t\"Helped me\"\t7\tMarch 5, 2016\t5\n"
)
TSV_CONTENT_B = (
    "\tdrugName\tcondition\treview\trating\tdate\tusefulCount\n"
    "2\tDrug C\tAlcohol Use Disorder\t\"Life saver\"\t10\tJune 3, 2017\t20\n"
)

@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "drugsComTrain_raw.tsv").write_text(TSV_CONTENT_A)
    (tmp_path / "drugsComTest_raw.tsv").write_text(TSV_CONTENT_B)
    return str(tmp_path)

def test_load_raw_merges_train_test(raw_dir):
    config = {"data": {"raw_dir": raw_dir,
                       "train_file": "drugsComTrain_raw.tsv",
                       "test_file": "drugsComTest_raw.tsv"}}
    df = load_raw(config)
    assert len(df) == 3
    assert set(df.columns) >= {"drugName", "condition", "review", "rating", "date", "usefulCount"}

def test_load_raw_resets_index(raw_dir):
    config = {"data": {"raw_dir": raw_dir,
                       "train_file": "drugsComTrain_raw.tsv",
                       "test_file": "drugsComTest_raw.tsv"}}
    df = load_raw(config)
    assert list(df.index) == list(range(len(df)))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ingest.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `load_raw` not defined yet.

- [ ] **Step 3: Implement `load_raw` in `src/ingest.py`**

```python
"""Data ingestion: loading, filtering, temporal feature extraction."""
import os
import sys
import pandas as pd
import yaml


def load_raw(config: dict) -> pd.DataFrame:
    """Load and merge train/test TSV files into a single DataFrame."""
    raw_dir = config["data"]["raw_dir"]
    train_path = os.path.join(raw_dir, config["data"]["train_file"])
    test_path = os.path.join(raw_dir, config["data"]["test_file"])
    train = pd.read_csv(train_path, sep="\t", index_col=0)
    test = pd.read_csv(test_path, sep="\t", index_col=0)
    df = pd.concat([train, test], ignore_index=True)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_ingest.py::test_load_raw_merges_train_test tests/test_ingest.py::test_load_raw_resets_index -v
```

Expected: 2 passed.

---

## Task 3: Addiction Filter (`src/ingest.py`)

**Files:**
- Modify: `src/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ingest.py`:

```python
from src.ingest import filter_addiction_related

FILTER_CONFIG = {
    "filter": {
        "addiction_conditions": ["opiate dependence", "alcohol use disorder"],
        "addiction_drugs": ["buprenorphine", "naltrexone"],
    }
}

def make_df(rows):
    return pd.DataFrame(rows, columns=["drugName", "condition", "review", "rating", "date", "usefulCount"])

def test_filter_includes_condition_match():
    df = make_df([
        ["Drug X", "Opiate Dependence", "review text", 8, "Jan 1, 2015", 5],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1

def test_filter_includes_drug_match():
    df = make_df([
        ["Buprenorphine", "Back Pain", "review text", 6, "Jan 1, 2015", 2],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1

def test_filter_excludes_unrelated():
    df = make_df([
        ["Aspirin", "Headache", "review text", 7, "Jan 1, 2015", 1],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 0

def test_filter_deduplicates_union():
    """A row matching both condition AND drug should appear only once."""
    df = make_df([
        ["Buprenorphine", "Opiate Dependence", "review text", 9, "Jan 1, 2015", 3],
    ])
    result = filter_addiction_related(df, FILTER_CONFIG)
    assert len(result) == 1
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_ingest.py::test_filter_includes_condition_match -v
```

Expected: `ImportError` — `filter_addiction_related` not defined.

- [ ] **Step 3: Implement `filter_addiction_related`**

Add to `src/ingest.py`:

```python
def filter_addiction_related(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep rows matching addiction conditions (substring) OR addiction drugs (exact, case-insensitive)."""
    conditions = [c.lower() for c in config["filter"]["addiction_conditions"]]
    drugs = [d.lower() for d in config["filter"]["addiction_drugs"]]

    cond_col = df["condition"].fillna("").str.lower()
    drug_col = df["drugName"].fillna("").str.lower()

    condition_mask = cond_col.apply(lambda x: any(c in x for c in conditions))
    drug_mask = drug_col.isin(drugs)

    return df[condition_mask | drug_mask].reset_index(drop=True)
```

- [ ] **Step 4: Run all filter tests**

```bash
pytest tests/test_ingest.py -k "filter" -v
```

Expected: 4 passed.

---

## Task 4: Temporal Features (`src/ingest.py`)

**Files:**
- Modify: `src/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_ingest.py`:

```python
from src.ingest import extract_temporal_features

def test_temporal_features_parse_date():
    df = make_df([
        ["Drug A", "Opiate Dependence", "text", 8, "October 6, 2013", 4],
        ["Drug B", "Alcohol Use Disorder", "text", 5, "March 15, 2016", 2],
    ])
    result = extract_temporal_features(df)
    assert "year" in result.columns
    assert "quarter" in result.columns
    assert "year_quarter" in result.columns
    assert result.iloc[0]["year"] == 2013
    assert result.iloc[0]["quarter"] == 4
    assert result.iloc[0]["year_quarter"] == "2013-Q4"
    assert result.iloc[1]["year_quarter"] == "2016-Q1"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_ingest.py::test_temporal_features_parse_date -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `extract_temporal_features`**

Add to `src/ingest.py`:

```python
def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date column and add year, quarter, year_quarter columns."""
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df["year"] = df["date_parsed"].dt.year
    df["quarter"] = df["date_parsed"].dt.quarter
    df["year_quarter"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_ingest.py::test_temporal_features_parse_date -v
```

Expected: 1 passed.

---

## Task 5: EDA + Pipeline Entry Point (`src/ingest.py`)

**Files:**
- Modify: `src/ingest.py`

- [ ] **Step 1: Add `run_eda` and `__main__` block**

Add to `src/ingest.py`:

```python
def run_eda(df: pd.DataFrame) -> None:
    """Print exploratory statistics to stderr."""
    print(f"\n=== EDA ===", file=sys.stderr)
    print(f"Total reviews: {len(df)}", file=sys.stderr)
    print(f"Date range: {df['date_parsed'].min().date()} to {df['date_parsed'].max().date()}", file=sys.stderr)
    print(f"Unique drugs: {df['drugName'].nunique()}", file=sys.stderr)
    print(f"Unique conditions: {df['condition'].nunique()}", file=sys.stderr)
    print(f"Year-quarter range: {df['year_quarter'].min()} to {df['year_quarter'].max()}", file=sys.stderr)
    print(f"\nTop 10 drugs by review count:", file=sys.stderr)
    print(df["drugName"].value_counts().head(10).to_string(), file=sys.stderr)
    print(f"\nTop conditions:", file=sys.stderr)
    print(df["condition"].value_counts().head(10).to_string(), file=sys.stderr)
    print(f"\nRating distribution:\n{df['rating'].describe()}", file=sys.stderr)


if __name__ == "__main__":
    import yaml
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
```

- [ ] **Step 2: Download the Kaggle dataset**

```bash
# Option A: kaggle CLI
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json first
kaggle datasets download jessicali9530/kuc-hackathon-winter-2018 -p data/raw --unzip

# Option B: manual download from
# https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
# Place drugsComTrain_raw.tsv and drugsComTest_raw.tsv in data/raw/
```

- [ ] **Step 3: Run the ingest pipeline**

```bash
python -m src.ingest
```

Expected output (stderr): EDA stats, line "Saved to data/processed/reviews_filtered.parquet". After filter, expect roughly 5,000–20,000 rows.

- [ ] **Step 4: Run all ingest tests**

```bash
pytest tests/test_ingest.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: data ingestion with addiction filter and temporal features"
```

---

## Task 6: Text Preprocessing (`src/preprocess.py`)

**Files:**
- Create: `src/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocess.py
import pandas as pd
import pytest
from src.preprocess import clean_text, preprocess_reviews

def test_clean_text_html_apostrophe():
    assert clean_text("It&#039;s great") == "it's great"

def test_clean_text_html_amp():
    assert clean_text("pain &amp; anxiety") == "pain & anxiety"

def test_clean_text_html_quote():
    assert clean_text("&quot;miracle drug&quot;") == '"miracle drug"'

def test_clean_text_extra_whitespace():
    assert clean_text("  too   many   spaces  ") == "too many spaces"

def test_clean_text_lowercases():
    assert clean_text("GREAT Drug") == "great drug"

def test_clean_text_strips_span_tags():
    assert clean_text("condition</span>") == "condition"

def test_preprocess_reviews_adds_column():
    df = pd.DataFrame({"review": ["It&#039;s great", "Pain &amp; anxiety"]})
    result = preprocess_reviews(df)
    assert "clean_review" in result.columns
    assert result.iloc[0]["clean_review"] == "it's great"
    assert result.iloc[1]["clean_review"] == "pain & anxiety"

def test_preprocess_reviews_preserves_original():
    df = pd.DataFrame({"review": ["It&#039;s great"]})
    result = preprocess_reviews(df)
    assert "review" in result.columns  # original preserved
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_preprocess.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/preprocess.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_preprocess.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Run preprocess on the filtered dataset**

```bash
python -m src.preprocess
```

Expected: "Preprocessing complete."

- [ ] **Step 6: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: text preprocessing with HTML decoding and normalization"
```

---

## Task 7: Embedding Generation (`src/embeddings.py`)

**Files:**
- Create: `src/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embeddings.py
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_embeddings.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/embeddings.py`**

```python
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
    return embeddings.astype(np.float32)


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
    import os
    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    embeddings, _ = load_or_generate_embeddings(df, config)
    print(f"Embeddings shape: {embeddings.shape}", file=sys.stderr)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_embeddings.py -v
```

Expected: 3 passed. Note: `test_generate_embeddings_shape` downloads the model on first run (~80MB).

- [ ] **Step 5: Run embedding generation**

```bash
python -m src.embeddings
```

Expected: progress bar, then "Embeddings shape: (N, 384)".

- [ ] **Step 6: Commit**

```bash
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat: sentence embedding generation with disk cache"
```

---

## Task 8: UMAP + HDBSCAN Clustering (`src/clustering.py`)

**Files:**
- Create: `src/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_clustering.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from unittest.mock import patch, MagicMock
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
}

def test_reduce_dimensions_output_shape():
    rng = np.random.default_rng(42)
    embeddings = rng.random((100, 50)).astype(np.float32)
    reduced = reduce_dimensions(embeddings, n_components=5, config=CONFIG)
    assert reduced.shape == (100, 5)

def test_hdbscan_returns_valid_labels():
    X, _ = make_blobs(n_samples=200, centers=4, n_features=5, random_state=42)
    labels = cluster_hdbscan(X.astype(np.float32), CONFIG)
    assert len(labels) == 200
    assert all(l >= -1 for l in labels)

def test_kmeans_returns_n_clusters():
    X, _ = make_blobs(n_samples=150, centers=3, n_features=5, random_state=42)
    labels = cluster_kmeans(X.astype(np.float32), n_clusters=3, config=CONFIG)
    assert len(labels) == 150
    assert len(set(labels)) == 3
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_clustering.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `reduce_dimensions`, `cluster_hdbscan`, `cluster_kmeans`**

```python
"""Dimensionality reduction and clustering for recovery stage discovery."""
import sys
import json
import numpy as np
import pandas as pd
import umap
import hdbscan
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
    return reducer.fit_transform(embeddings).astype(np.float32)


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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_clustering.py::test_reduce_dimensions_output_shape tests/test_clustering.py::test_hdbscan_returns_valid_labels tests/test_clustering.py::test_kmeans_returns_n_clusters -v
```

Expected: 3 passed.

---

## Task 9: LLM Stage Labeling (`src/clustering.py`)

**Files:**
- Modify: `src/clustering.py`
- Modify: `tests/test_clustering.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_clustering.py`:

```python
from src.clustering import label_clusters_with_llm, assign_cluster_labels

MOCK_LLM_RESPONSE = json.dumps({
    "stage_name": "SEEKING_STABILITY",
    "ttm_stage": "Action",
    "description": "Patients newly on medication, cautiously optimistic.",
    "risk_level": "MODERATE"
})

def test_label_clusters_with_llm_mock():
    import json
    rng = np.random.default_rng(0)
    n = 30
    embeddings = rng.random((n, 384)).astype(np.float32)
    df = pd.DataFrame({
        "clean_review": [f"review {i}" for i in range(n)],
        "drugName": ["Suboxone"] * n,
    })
    cluster_labels = np.array([0] * 15 + [1] * 15)

    mock_response = MagicMock()
    mock_response.message.content = MOCK_LLM_RESPONSE

    with patch("src.clustering.ollama.chat", return_value=mock_response):
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_clustering.py::test_label_clusters_with_llm_mock tests/test_clustering.py::test_assign_cluster_labels -v
```

Expected: `ImportError` for `label_clusters_with_llm`.

- [ ] **Step 3: Add import and implement `label_clusters_with_llm` and `assign_cluster_labels`**

Add to top of `src/clustering.py`:

```python
import json
import ollama
```

Add functions to `src/clustering.py`:

```python
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

        # Centroid and cosine similarity
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
            parsed = json.loads(response.message.content)
        except json.JSONDecodeError:
            parsed = {
                "stage_name": f"CLUSTER_{cluster_id}",
                "ttm_stage": "Unknown",
                "description": "Could not parse LLM response.",
                "risk_level": "MODERATE",
            }
        results[cluster_id] = parsed

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
        cid = row["cluster_id"]
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
```

- [ ] **Step 4: Run all clustering tests**

```bash
pytest tests/test_clustering.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Add `__main__` block and run the clustering pipeline**

Add to `src/clustering.py`:

```python
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
    reduced_5d = reduce_dimensions(embeddings, n_components=config["clustering"]["umap_n_components_cluster"], config=config)

    print("Reducing dimensions (2-dim for visualization)...", file=sys.stderr)
    reduced_2d = reduce_dimensions(embeddings, n_components=config["clustering"]["umap_n_components_viz"], config=config)

    print("Running HDBSCAN...", file=sys.stderr)
    hdbscan_labels = cluster_hdbscan(reduced_5d, config)
    n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    print(f"HDBSCAN found {n_clusters} clusters, {(hdbscan_labels == -1).sum()} noise points", file=sys.stderr)

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
```

```bash
python -m src.clustering
```

Expected: UMAP progress, HDBSCAN output, LLM labeling per cluster, "Clustering complete."

- [ ] **Step 6: Commit**

```bash
git add src/clustering.py tests/test_clustering.py
git commit -m "feat: UMAP+HDBSCAN clustering with LLM stage labeling"
```

---

## Task 10: Keyword Baseline Classifier (`src/baseline.py`)

**Files:**
- Create: `src/baseline.py`
- Create: `tests/test_baseline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_baseline.py
import pandas as pd
import pytest
from src.baseline import classify_risk_baseline, batch_classify_baseline

CONFIG = {
    "baseline": {
        "high_risk_patterns": ["relapsed", "relapse", "overdose", "hopeless"],
        "low_risk_patterns": ["sober", "sobriety", "changed my life", "in recovery"],
    }
}

def test_high_risk_detection():
    assert classify_risk_baseline("i relapsed after three months", CONFIG) == "HIGH"

def test_low_risk_detection():
    assert classify_risk_baseline("i have been sober for six months", CONFIG) == "LOW"

def test_moderate_default():
    assert classify_risk_baseline("the medication made me nauseous", CONFIG) == "MODERATE"

def test_high_takes_priority_over_low():
    # HIGH patterns take precedence
    assert classify_risk_baseline("relapsed but now in recovery", CONFIG) == "HIGH"

def test_batch_classify_adds_column():
    df = pd.DataFrame({"clean_review": [
        "i relapsed last week",
        "sober for a year now",
        "just a regular review",
    ]})
    result = batch_classify_baseline(df, CONFIG)
    assert "baseline_risk" in result.columns
    assert result.iloc[0]["baseline_risk"] == "HIGH"
    assert result.iloc[1]["baseline_risk"] == "LOW"
    assert result.iloc[2]["baseline_risk"] == "MODERATE"
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_baseline.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/baseline.py`**

```python
"""Keyword/regex baseline risk classifier."""
import re
import sys
import pandas as pd


def classify_risk_baseline(text: str, config: dict) -> str:
    """Classify a single review as HIGH, LOW, or MODERATE using keyword rules.

    HIGH takes priority over LOW.
    """
    text_lower = text.lower()
    for pattern in config["baseline"]["high_risk_patterns"]:
        if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
            return "HIGH"
    for pattern in config["baseline"]["low_risk_patterns"]:
        if re.search(r"\b" + re.escape(pattern) + r"\b", text_lower):
            return "LOW"
    return "MODERATE"


def batch_classify_baseline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add baseline_risk column to df."""
    df = df.copy()
    df["baseline_risk"] = df["clean_review"].apply(lambda t: classify_risk_baseline(t, config))
    return df


if __name__ == "__main__":
    import os
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    df = batch_classify_baseline(df, config)
    print(df["baseline_risk"].value_counts().to_string(), file=sys.stderr)
    df.to_parquet(parquet_path, index=False)
    print("Baseline classification saved.", file=sys.stderr)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_baseline.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/baseline.py tests/test_baseline.py
git commit -m "feat: keyword baseline risk classifier"
```

---

## Task 11: LLM Risk Classifier (`src/classify.py`)

**Files:**
- Create: `src/classify.py`
- Create: `tests/test_classify.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_classify.py
import pandas as pd
import pytest
import json
from unittest.mock import patch, MagicMock
from src.classify import classify_with_llm, batch_classify_llm

CONFIG = {
    "llm": {"model": "llama3.1:8b", "batch_size": 2, "temperature": 0.1}
}

MOCK_RESPONSE_JSON = json.dumps({
    "stage_label": "SEEKING_STABILITY",
    "risk_level": "MODERATE",
    "rationale": "Patient is newly on medication and cautiously optimistic."
})

def make_mock_response(content: str):
    m = MagicMock()
    m.message.content = content
    return m

def test_classify_with_llm_returns_required_keys():
    with patch("src.classify.ollama.chat", return_value=make_mock_response(MOCK_RESPONSE_JSON)):
        results = classify_with_llm(["i just started suboxone"], CONFIG)
    assert len(results) == 1
    assert "stage_label" in results[0]
    assert "risk_level" in results[0]
    assert "rationale" in results[0]

def test_classify_with_llm_handles_parse_error():
    with patch("src.classify.ollama.chat", return_value=make_mock_response("not json")):
        results = classify_with_llm(["some review"], CONFIG)
    assert results[0]["risk_level"] == "MODERATE"

def test_batch_classify_llm_adds_columns():
    df = pd.DataFrame({"clean_review": ["review one", "review two"]})
    with patch("src.classify.ollama.chat", return_value=make_mock_response(MOCK_RESPONSE_JSON)):
        result = batch_classify_llm(df, CONFIG)
    assert "llm_risk" in result.columns
    assert "llm_stage" in result.columns
    assert "llm_rationale" in result.columns
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_classify.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/classify.py`**

```python
"""LLM-based risk classification with stage labels and rationale."""
import json
import sys
import pandas as pd
import ollama
from tqdm import tqdm

CLASSIFY_PROMPT = """You are a public health data analyst. Read this patient review of an addiction treatment medication.
Classify the review into a recovery stage and risk level.

Review: "{review}"

Respond with ONLY valid JSON:
{{
  "stage_label": "SHORT_STAGE_NAME (e.g. SEEKING_STABILITY, RELAPSE_RISK, IN_RECOVERY, EARLY_TREATMENT, MAINTENANCE)",
  "risk_level": "HIGH, MODERATE, or LOW",
  "rationale": "One sentence explaining the classification."
}}"""


def classify_with_llm(texts: list[str], config: dict) -> list[dict]:
    """Classify a list of review texts. Returns list of dicts with stage_label, risk_level, rationale."""
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]
    results = []

    for text in texts:
        prompt = CLASSIFY_PROMPT.format(review=text[:500])  # truncate to avoid context overflow
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        try:
            parsed = json.loads(response.message.content)
            if "risk_level" not in parsed:
                raise ValueError("Missing risk_level")
        except (json.JSONDecodeError, ValueError):
            parsed = {
                "stage_label": "UNKNOWN",
                "risk_level": "MODERATE",
                "rationale": "Could not parse LLM response.",
            }
        results.append(parsed)

    return results


def batch_classify_llm(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add llm_risk, llm_stage, llm_rationale columns. Processes in batches."""
    df = df.copy()
    batch_size = config["llm"]["batch_size"]
    all_results: list[dict] = []

    texts = df["clean_review"].tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc="LLM classification"):
        batch = texts[i : i + batch_size]
        all_results.extend(classify_with_llm(batch, config))

    df["llm_risk"] = [r["risk_level"] for r in all_results]
    df["llm_stage"] = [r.get("stage_label", "UNKNOWN") for r in all_results]
    df["llm_rationale"] = [r.get("rationale", "") for r in all_results]
    return df


if __name__ == "__main__":
    import os
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    print(f"Classifying {len(df)} reviews with LLM...", file=sys.stderr)
    df = batch_classify_llm(df, config)
    df.to_parquet(parquet_path, index=False)
    print(df["llm_risk"].value_counts().to_string(), file=sys.stderr)
    print("LLM classification saved.", file=sys.stderr)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_classify.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run LLM classification on full dataset**

```bash
python -m src.classify
```

Expected: tqdm progress bar, risk distribution printed. Note: this will take significant time (~10–30 min for 10k reviews at local Ollama speeds). Consider running overnight or using a smaller sample for development.

- [ ] **Step 6: Commit**

```bash
git add src/classify.py tests/test_classify.py
git commit -m "feat: LLM-based risk classification with stage labels and rationale"
```

---

## Task 12: Approach Evaluation (`src/evaluate.py`)

**Files:**
- Create: `src/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluate.py
import pandas as pd
import pytest
from src.evaluate import compute_metrics, compare_approaches, save_comparison

def test_compute_metrics_perfect_prediction():
    y_true = ["HIGH", "LOW", "MODERATE", "HIGH"]
    y_pred = ["HIGH", "LOW", "MODERATE", "HIGH"]
    metrics = compute_metrics(y_true, y_pred, labels=["HIGH", "LOW", "MODERATE"])
    assert metrics["HIGH"]["f1-score"] == pytest.approx(1.0)
    assert metrics["LOW"]["f1-score"] == pytest.approx(1.0)

def test_compute_metrics_partial():
    y_true = ["HIGH", "HIGH", "LOW", "MODERATE"]
    y_pred = ["HIGH", "LOW",  "LOW", "MODERATE"]
    metrics = compute_metrics(y_true, y_pred, labels=["HIGH", "LOW", "MODERATE"])
    assert 0.0 <= metrics["HIGH"]["f1-score"] <= 1.0

def test_compare_approaches_has_required_columns():
    df = pd.DataFrame({
        "llm_risk":      ["HIGH", "LOW", "MODERATE", "HIGH"],
        "baseline_risk": ["HIGH", "LOW", "HIGH",     "HIGH"],
        "risk_level":    ["HIGH", "LOW", "MODERATE", "LOW"],
    })
    result = compare_approaches(df)
    assert "approach" in result.columns
    assert set(result["approach"]) == {"baseline_vs_llm", "cluster_vs_llm"}
    assert "macro_f1" in result.columns
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/evaluate.py`**

```python
"""Evaluation: compare three classification approaches."""
import os
import sys
import pandas as pd
from sklearn.metrics import classification_report


LABELS = ["HIGH", "MODERATE", "LOW"]


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Return per-class precision/recall/F1 dict from classification_report."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return report


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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_evaluate.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run evaluation**

```bash
python -m src.evaluate
```

Expected: approach comparison table printed and saved.

- [ ] **Step 6: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: three-approach evaluation comparing baseline, cluster, and LLM classifiers"
```

---

## Task 13: Temporal Analysis (`src/temporal.py`)

**Files:**
- Create: `src/temporal.py`
- Create: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_temporal.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.temporal import (
    compute_stage_distribution,
    compute_drug_trends,
    detect_spikes,
    generate_spike_narratives,
)

def make_temporal_df():
    return pd.DataFrame({
        "year_quarter": ["2013-Q1"] * 10 + ["2013-Q2"] * 10 + ["2013-Q3"] * 10,
        "stage_name": (["RELAPSE_RISK"] * 7 + ["MAINTENANCE"] * 3) +
                      (["RELAPSE_RISK"] * 4 + ["MAINTENANCE"] * 6) +
                      (["RELAPSE_RISK"] * 5 + ["MAINTENANCE"] * 5),
        "risk_level": (["HIGH"] * 7 + ["LOW"] * 3) +
                      (["HIGH"] * 4 + ["LOW"] * 6) +
                      (["HIGH"] * 5 + ["LOW"] * 5),
        "drugName": ["Suboxone"] * 30,
        "rating": [7.0] * 30,
        "usefulCount": list(range(30)),
        "clean_review": [f"review {i}" for i in range(30)],
    })

def test_stage_distribution_sums_to_100():
    df = make_temporal_df()
    dist = compute_stage_distribution(df)
    row_sums = dist.sum(axis=1)
    for val in row_sums:
        assert abs(val - 100.0) < 0.01

def test_drug_trends_has_required_columns():
    df = make_temporal_df()
    trends = compute_drug_trends(df)
    assert "drugName" in trends.columns
    assert "year_quarter" in trends.columns
    assert "median_rating" in trends.columns
    assert "dominant_stage" in trends.columns

def test_detect_spikes_finds_known_spike():
    """Insert a clear spike at Q5 and verify it is detected."""
    quarters = [f"2012-Q{q}" for q in range(1, 5)] + ["2013-Q1"] + [f"2013-Q{q}" for q in range(2, 5)]
    high_counts = [5, 5, 5, 5, 50, 5, 5, 5]  # Q5 is a spike
    df_rows = []
    for q, count in zip(quarters, high_counts):
        for _ in range(count):
            df_rows.append({"year_quarter": q, "risk_level": "HIGH", "usefulCount": 1, "clean_review": "relapsed"})
        for _ in range(5):
            df_rows.append({"year_quarter": q, "risk_level": "LOW", "usefulCount": 1, "clean_review": "sober"})
    df = pd.DataFrame(df_rows)
    config = {"temporal": {"spike_window": 4, "spike_threshold": 2.0}}
    _, spike_quarters = detect_spikes(df, config)
    assert "2013-Q1" in spike_quarters

def test_generate_spike_narratives_mock():
    df = make_temporal_df()
    config = {"llm": {"model": "llama3.1:8b", "temperature": 0.1, "spike_sample_size": 5}}
    spike_quarters = ["2013-Q1"]

    mock_resp = MagicMock()
    mock_resp.message.content = "During Q1 2013, relapse language spiked. Reviews described medication access issues. This aligned with national opioid availability changes."

    with patch("src.temporal.ollama.chat", return_value=mock_resp):
        narratives = generate_spike_narratives(df, spike_quarters, config)

    assert "2013-Q1" in narratives
    assert len(narratives["2013-Q1"]) > 10
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_temporal.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/temporal.py`**

```python
"""Temporal analysis: stage drift, drug trends, spike detection, narratives, plots."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ollama


def compute_stage_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % of each stage per year_quarter.

    Returns DataFrame: index=year_quarter (sorted), columns=stage_names, values=% share.
    """
    counts = df.groupby(["year_quarter", "stage_name"]).size().unstack(fill_value=0)
    dist = counts.div(counts.sum(axis=1), axis=0) * 100
    dist.index = pd.CategoricalIndex(dist.index, ordered=True)
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

    high_df = df[df["risk_level"] == "HIGH"]
    counts = high_df.groupby("year_quarter").size().reset_index(name="high_count")
    counts = counts.sort_values("year_quarter").reset_index(drop=True)

    rolling_mean = counts["high_count"].rolling(window=window, min_periods=2).mean()
    rolling_std = counts["high_count"].rolling(window=window, min_periods=2).std()
    counts["z_score"] = (counts["high_count"] - rolling_mean) / rolling_std.replace(0, np.nan)
    counts["z_score"] = counts["z_score"].fillna(0.0)

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

    Returns dict mapping year_quarter -> narrative string.
    """
    model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]
    n_samples = config["llm"]["spike_sample_size"]
    narratives: dict[str, str] = {}

    for quarter in spike_quarters:
        quarter_df = df[
            (df["year_quarter"] == quarter) & (df["risk_level"] == "HIGH")
        ].sort_values("usefulCount", ascending=False).head(n_samples)

        reviews_text = "\n".join(
            f"- {row['clean_review'][:200]}" for _, row in quarter_df.iterrows()
        )
        prompt = SPIKE_NARRATIVE_PROMPT.format(
            n=len(quarter_df), quarter=quarter, reviews=reviews_text
        )
        print(f"Generating narrative for spike quarter {quarter}...", file=sys.stderr)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        narratives[quarter] = response.message.content.strip()

    return narratives


def plot_spike_detection(
    time_series: pd.DataFrame,
    spike_quarters: list[str],
    narratives: dict[str, str],
    config: dict,
) -> None:
    """Save spike detection hero chart to outputs/figures/spike_detection.png."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_series["year_quarter"], time_series["high_count"],
            color="#d62728", linewidth=2, marker="o", markersize=4, label="HIGH-risk reviews")

    for quarter in spike_quarters:
        idx = time_series[time_series["year_quarter"] == quarter].index
        if len(idx):
            ax.axvspan(idx[0] - 0.4, idx[0] + 0.4, alpha=0.2, color="#d62728")
            narrative = narratives.get(quarter, "")
            if narrative:
                short = ". ".join(narrative.split(".")[:1]) + "."
                ax.annotate(
                    f"↑ {quarter}\n{short[:60]}...",
                    xy=(idx[0], time_series.loc[idx[0], "high_count"]),
                    xytext=(idx[0] + 1, time_series.loc[idx[0], "high_count"] * 1.1),
                    fontsize=7, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728"),
                )

    tick_step = max(1, len(time_series) // 10)
    ax.set_xticks(range(0, len(time_series), tick_step))
    ax.set_xticklabels(time_series["year_quarter"].iloc[::tick_step], rotation=45, ha="right", fontsize=8)
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

    fig, ax = plt.subplots(figsize=(14, 6))
    for drug in top_drugs:
        drug_df = filtered[filtered["drugName"] == drug].sort_values("year_quarter")
        ax.plot(drug_df["year_quarter"], drug_df["median_rating"], label=drug, linewidth=1.5, marker=".")

    tick_step = max(1, len(drug_trends["year_quarter"].unique()) // 10)
    all_quarters = sorted(drug_trends["year_quarter"].unique())
    ax.set_xticks(range(0, len(all_quarters), tick_step))
    ax.set_xticklabels([all_quarters[i] for i in range(0, len(all_quarters), tick_step)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Median Patient Rating (1–10)")
    ax.set_title("Treatment Drug Effectiveness Trends Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(config["paths"]["figures"], "drug_trends.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved drug trends chart to {path}", file=sys.stderr)


def plot_umap_clusters(
    df: pd.DataFrame, cluster_stages: dict, config: dict
) -> None:
    """Save 2D UMAP scatter plot colored by recovery stage."""
    fig, ax = plt.subplots(figsize=(12, 8))
    stages = df["stage_name"].unique()
    palette = sns.color_palette("tab10", len(stages))
    color_map = dict(zip(stages, palette))

    for stage in stages:
        mask = df["stage_name"] == stage
        ax.scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                   c=[color_map[stage]], label=stage, s=5, alpha=0.6)

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
    plot_stage_drift(stage_dist, config)

    print("Computing drug trends...", file=sys.stderr)
    drug_trends = compute_drug_trends(df)
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_temporal.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run temporal analysis**

```bash
python -m src.temporal
```

Expected: 4 figures saved to `outputs/figures/`, spike narratives saved to `outputs/summaries/spike_narratives.json`.

- [ ] **Step 6: Commit**

```bash
git add src/temporal.py tests/test_temporal.py
git commit -m "feat: temporal analysis with spike detection, stage drift, and drug trends"
```

---

## Task 14: Explainability Summaries (`src/explain.py`)

**Files:**
- Create: `src/explain.py`

- [ ] **Step 1: Implement `src/explain.py`**

```python
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

        sample = cluster_df.sort_values("usefulCount", ascending=False).head(5)
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
        sections.append(response.message.content.strip())
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
```

- [ ] **Step 2: Run explainability**

```bash
python -m src.explain
```

Expected: one summary per cluster printed to stderr, `outputs/summaries/cluster_summaries.md` created.

- [ ] **Step 3: Commit**

```bash
git add src/explain.py
git commit -m "feat: LLM-generated cluster summaries for public health analyst audience"
```

---

## Task 15: Streamlit Dashboard (`app.py`)

**Files:**
- Create: `app.py`

- [ ] **Step 1: Implement `app.py`**

```python
"""Streamlit dashboard: Recovery Trajectory Staging System."""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from src.temporal import compute_stage_distribution, compute_drug_trends, detect_spikes

st.set_page_config(page_title="High Alert: Recovery Trajectory Staging", layout="wide")


@st.cache_data
def load_data():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    parquet_path = os.path.join(config["data"]["processed_dir"], config["data"]["filtered_file"])
    df = pd.read_parquet(parquet_path)
    return df, config


@st.cache_data
def load_cluster_stages():
    with open("outputs/tables/cluster_stages.json") as f:
        return json.load(f)


@st.cache_data
def load_spike_narratives():
    path = "outputs/summaries/spike_narratives.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_cluster_summaries():
    path = "outputs/summaries/cluster_summaries.md"
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return "Summaries not yet generated. Run `python -m src.explain` first."


df, config = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Recovery Stages", "Temporal Analysis", "Drug Comparison"])

# --- Tab 1: Overview ---
with tab1:
    st.title("High Alert: Substance Abuse Risk Detection")
    st.markdown("**Recovery Trajectory Staging from Addiction Treatment Reviews**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Unique Drugs", df["drugName"].nunique())
    col3.metric("Date Range", f"{df['year'].min():.0f}–{df['year'].max():.0f}")
    col4.metric("Recovery Stages", df["stage_name"].nunique())

    st.subheader("Risk Level Distribution")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {"HIGH": "#d62728", "MODERATE": "#ff7f0e", "LOW": "#2ca02c"}
    ax.bar(risk_counts["risk_level"], risk_counts["count"],
           color=[colors.get(r, "#1f77b4") for r in risk_counts["risk_level"]])
    ax.set_ylabel("Review Count")
    st.pyplot(fig)
    plt.close()

    st.subheader("Approach Comparison")
    comp_path = "outputs/tables/approach_comparison.csv"
    if os.path.exists(comp_path):
        st.dataframe(pd.read_csv(comp_path), use_container_width=True)
    else:
        st.info("Run `python -m src.evaluate` to generate approach comparison.")

# --- Tab 2: Recovery Stages ---
with tab2:
    st.title("Recovery Stage Discovery")
    st.markdown("Stages discovered via HDBSCAN clustering, labeled by LLM.")

    if "umap_x" in df.columns:
        st.subheader("2D UMAP Projection by Recovery Stage")
        stages = df["stage_name"].unique()
        palette = sns.color_palette("tab10", len(stages))
        color_map = dict(zip(stages, [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in palette]))

        fig, ax = plt.subplots(figsize=(10, 7))
        for stage in stages:
            mask = df["stage_name"] == stage
            ax.scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                       c=[color_map[stage]], label=stage, s=5, alpha=0.6)
        ax.legend(fontsize=8, markerscale=3)
        ax.set_title("Recovery Stage Clusters (UMAP 2D)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run `python -m src.clustering` to generate UMAP coordinates.")

    st.subheader("Cluster Summaries (Public Health Analyst View)")
    st.markdown(load_cluster_summaries())

# --- Tab 3: Temporal Analysis ---
with tab3:
    st.title("Temporal Risk Signal Analysis")

    spike_path = "outputs/figures/spike_detection.png"
    if os.path.exists(spike_path):
        st.subheader("HIGH-Risk Review Spike Detection (Hero Chart)")
        st.image(spike_path, use_column_width=True)

        narratives = load_spike_narratives()
        if narratives:
            st.subheader("Spike Narratives")
            for quarter, narrative in narratives.items():
                with st.expander(f"Spike: {quarter}"):
                    st.write(narrative)
    else:
        st.info("Run `python -m src.temporal` to generate charts.")

    drift_path = "outputs/figures/stage_drift.png"
    if os.path.exists(drift_path):
        st.subheader("Recovery Stage Distribution Over Time")
        st.image(drift_path, use_column_width=True)

# --- Tab 4: Drug Comparison ---
with tab4:
    st.title("Treatment Drug Effectiveness Trends")

    drug_path = "outputs/tables/drug_trends.csv"
    if os.path.exists(drug_path):
        drug_trends = pd.read_csv(drug_path)
        top_drugs = (
            drug_trends.groupby("drugName")["review_count"].sum()
            .nlargest(20).index.tolist()
        )
        selected_drugs = st.multiselect("Select drugs to compare:", top_drugs, default=top_drugs[:5])

        if selected_drugs:
            filtered = drug_trends[drug_trends["drugName"].isin(selected_drugs)]
            fig, ax = plt.subplots(figsize=(12, 5))
            for drug in selected_drugs:
                d = filtered[filtered["drugName"] == drug].sort_values("year_quarter")
                ax.plot(d["year_quarter"], d["median_rating"], label=drug, linewidth=1.5, marker=".")
            all_q = sorted(filtered["year_quarter"].unique())
            step = max(1, len(all_q) // 10)
            ax.set_xticks(range(0, len(all_q), step))
            ax.set_xticklabels([all_q[i] for i in range(0, len(all_q), step)], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Median Rating (1–10)")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()

            st.subheader("Raw Trends Data")
            st.dataframe(filtered.sort_values(["drugName", "year_quarter"]), use_container_width=True)
    else:
        st.info("Run `python -m src.temporal` to generate drug trends data.")
```

- [ ] **Step 2: Run the dashboard**

```bash
streamlit run app.py
```

Expected: browser opens with 4-tab dashboard. Verify all tabs render without errors. Tabs with missing data show info messages rather than crashing.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: streamlit dashboard with 4-tab view of recovery staging system"
```

---

## Task 16: `reproduce.sh` and README

**Files:**
- Create: `reproduce.sh`
- Create: `README.md`

- [ ] **Step 1: Create `reproduce.sh`**

```bash
#!/usr/bin/env bash
# reproduce.sh — run the full High Alert pipeline from scratch
set -euo pipefail

echo "=== High Alert: Recovery Trajectory Staging Pipeline ==="
echo ""

echo "[1/6] Ingesting and filtering data..."
python -m src.ingest

echo "[2/6] Preprocessing text..."
python -m src.preprocess

echo "[3/6] Generating embeddings..."
python -m src.embeddings

echo "[4/6] Clustering and labeling recovery stages..."
python -m src.clustering

echo "[5/6] Running risk classification (baseline + LLM)..."
python -m src.baseline
python -m src.classify
python -m src.evaluate

echo "[6/6] Temporal analysis and explainability..."
python -m src.temporal
python -m src.explain

echo ""
echo "=== Pipeline complete. ==="
echo "Outputs saved to outputs/"
echo "Launch dashboard: streamlit run app.py"
```

```bash
chmod +x reproduce.sh
```

- [ ] **Step 2: Create `README.md`**

```markdown
# High Alert: Recovery Trajectory Staging System

AI-driven pipeline for detecting substance abuse risk signals from addiction treatment drug reviews. Discovers recovery stages from patient language using HDBSCAN clustering, validates them against the Transtheoretical Model (TTM), and surfaces temporal risk signals for public health analysts.

**NRT AI Challenge — CS 5542 Big Data Analytics | UMKC Spring 2026**

## Novel Contribution

Instead of predefined binary risk labels, this system *discovers* recovery stages from patient language, then labels them with an LLM. The result: data-driven population-level recovery staging with temporal spike detection.

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally with `llama3.1:8b`:
  ```bash
  ollama pull llama3.1:8b
  ```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download dataset
```bash
# Option A: kaggle CLI
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json
kaggle datasets download jessicali9530/kuc-hackathon-winter-2018 -p data/raw --unzip

# Option B: Manual download
# https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
# Place drugsComTrain_raw.tsv and drugsComTest_raw.tsv in data/raw/
```

## Run

### Full pipeline (single command)
```bash
./reproduce.sh
```

### Individual modules
```bash
python -m src.ingest       # Load, filter, save parquet
python -m src.preprocess   # Clean text
python -m src.embeddings   # Generate / load cached embeddings
python -m src.clustering   # UMAP + HDBSCAN + LLM stage labeling
python -m src.baseline     # Keyword baseline classifier
python -m src.classify     # LLM classifier
python -m src.evaluate     # Approach comparison metrics
python -m src.temporal     # Temporal analysis + spike detection + plots
python -m src.explain      # Cluster summaries for analysts
```

### Dashboard
```bash
streamlit run app.py
```

## Tests
```bash
pytest tests/ -v
```

## Pipeline Architecture
```
Raw CSV → Ingest+Filter → Preprocess → Embed → Cluster → Stage Label
                                                          ↓
                                               Three-Approach Comparison
                                               (baseline / cluster / LLM)
                                                          ↓
                                               Temporal Analysis + Spikes
                                                          ↓
                                               Explainability Summaries
                                                          ↓
                                               Streamlit Dashboard
```

## Outputs
| File | Description |
|------|-------------|
| `outputs/figures/spike_detection.png` | Hero chart: HIGH-risk spikes over time |
| `outputs/figures/stage_drift.png` | Stage distribution drift (stacked area) |
| `outputs/figures/drug_trends.png` | Drug effectiveness trends |
| `outputs/figures/umap_clusters.png` | 2D UMAP cluster visualization |
| `outputs/tables/approach_comparison.csv` | Baseline vs cluster vs LLM metrics |
| `outputs/tables/cluster_stages.json` | Discovered stage labels + TTM mapping |
| `outputs/summaries/cluster_summaries.md` | Analyst-facing cluster summaries |
| `outputs/summaries/spike_narratives.json` | LLM narratives per spike quarter |

## Ethics
- No PII. Population-level insights only.
- Dataset is anonymized patient reviews with no re-identification.
- LLM outputs framed as population-level trends, not individual diagnosis.

## Team
- Ben Blake (MSCS, UMKC)
```

- [ ] **Step 3: Run all tests one final time**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Final commit**

```bash
git add reproduce.sh README.md
git commit -m "feat: reproduce.sh pipeline script and README"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Data layer: condition ∪ drug filter — Task 3
- ✅ Embedding + caching — Task 7, 8
- ✅ UMAP 5-dim cluster + 2-dim viz — Task 8
- ✅ HDBSCAN primary, KMeans secondary — Task 8
- ✅ Stage labeling via LLM (cosine similarity centroid) — Task 9
- ✅ Three approaches (keyword / cluster / LLM) all producing risk_level — Tasks 10, 11, 12
- ✅ Evaluation with LLM as pseudo-ground-truth — Task 12
- ✅ Stage distribution drift — Task 13
- ✅ Drug trends — Task 13
- ✅ Spike detection with z-score — Task 13
- ✅ Spike narratives (top-20 by usefulCount) — Task 13
- ✅ Cluster summaries for analyst audience — Task 14
- ✅ Streamlit dashboard (4 tabs) — Task 15
- ✅ reproduce.sh + README — Task 16
- ✅ config.yaml driven — Task 1
- ✅ Ethical framing in prompts and README — Throughout
