# High Alert: Recovery Trajectory Staging System

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

**Application**: [https://high-alert.streamlit.app](https://high-alert.streamlit.app)
**Video**: [https://vimeo.com/1180195741](https://vimeo.com/1180195741)

AI-driven pipeline for detecting substance abuse risk signals from addiction treatment drug reviews. Discovers recovery stages from patient language using HDBSCAN clustering, validates them against the Transtheoretical Model (TTM), and surfaces temporal risk signals for public health analysts.

**NRT AI Challenge — CS 5542 Big Data Analytics | UMKC Spring 2026**

## Novel Contribution

Instead of predefined binary risk labels, this system *discovers* recovery stages from patient language, then labels them with an LLM. The result: data-driven population-level recovery staging with temporal spike detection.

## Setup

### Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com) for LLM inference

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure API key
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### Download dataset
```bash
# Option A: kaggle CLI
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json
kaggle datasets download jessicali9530/kuc-hackathon-winter-2018 -p data/raw --unzip

# Option B: Manual download
# https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
# Place drugsComTrain_raw.csv and drugsComTest_raw.csv in data/raw/
```

## Run

### Full pipeline (single command)
```bash
./reproduce.sh
```

### Individual modules
```bash
.venv/bin/python -m src.ingest       # Load, filter, save parquet
.venv/bin/python -m src.preprocess   # Clean text
.venv/bin/python -m src.embeddings   # Generate / load cached embeddings
.venv/bin/python -m src.clustering   # UMAP + HDBSCAN + LLM stage labeling
.venv/bin/python -m src.baseline     # Keyword baseline classifier
.venv/bin/python -m src.classify     # LLM classifier
.venv/bin/python -m src.evaluate     # Approach comparison metrics
.venv/bin/python -m src.temporal     # Temporal analysis + spike detection + plots
.venv/bin/python -m src.explain      # Cluster summaries for analysts
```

### Dashboard
```bash
.venv/bin/streamlit run app.py
```

## Tests
```bash
.venv/bin/pytest tests/ -v
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
- Ben Blake <ben.blake@umkc.edu>
