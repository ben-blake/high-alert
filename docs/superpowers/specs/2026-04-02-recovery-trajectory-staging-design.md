# Design: Recovery Trajectory Staging System
**Date:** 2026-04-02  
**Project:** NRT AI Challenge — Substance Abuse Risk Detection  
**Track:** A (AI Modeling and Reasoning)  
**Deadline:** April 6, 2026, 12:00 PM

---

## Novel Contribution

Most challenge submissions will do binary risk detection ("does this post mention substance use?"). This system instead discovers recovery stages from patient language — without pre-defining them — then validates those stages against the Transtheoretical Model (TTM). The claim: we can observe population-level shifts in treatment experience from review text alone, including detecting public health crisis signals.

---

## Architecture

```
Raw CSV (KUC Kaggle dataset)
  │
  ▼
Ingest + Filter (condition ∪ drug name) → ~5–20k reviews
  │
  ▼
Preprocess (clean text, extract temporal features)
  │
  ▼
Embed (all-MiniLM-L6-v2 → 384-dim, cached to disk)
  │
  ▼
UMAP (5-dim) → HDBSCAN → discover N clusters
       └── UMAP (2-dim) → scatter plot
  │
  ▼
LLM labels each cluster → stage_name + TTM mapping + description
  │
  ├──► Three-approach comparison (baseline / embedding / LLM) → risk_level
  │
  ├──► Temporal analysis (stage drift + spike detection + drug trends)
  │
  └──► Explainability summaries (per cluster + per spike)
        │
        └──► Streamlit dashboard + static outputs
```

---

## Data Layer

**Source:** KUC Hackathon Winter 2018 — `drugsComTrain_raw.csv` + `drugsComTest_raw.csv`, merged.

**Filter — condition ∪ drug name (both lists in `config.yaml`):**

Addiction conditions (case-insensitive substring match):
- opiate dependence, opioid dependence, alcohol use disorder,
  benzodiazepine withdrawal, cocaine dependence, substance abuse,
  drug dependence, heroin dependence, opioid withdrawal,
  alcohol withdrawal, nicotine dependence, methamphetamine dependence

Addiction drugs (case-insensitive exact match on `drugName`):
- buprenorphine, methadone, naltrexone, suboxone, vivitrol,
  disulfiram, acamprosate, naloxone, subutex, zubsolv,
  varenicline, chantix, campral

**Temporal features extracted at ingest:**
- `date` → parsed datetime
- `year`, `quarter`, `year_quarter` (e.g. "2013-Q3")

**Output:** `data/processed/reviews_filtered.parquet`

---

## Embedding + Clustering

**Embeddings:**
- Model: `all-MiniLM-L6-v2` (384-dim)
- Cache: `data/processed/embeddings.npy` + `data/processed/embedding_index.json`
- Skip regeneration if cache exists and row count matches

**Dimensionality reduction:**
- UMAP → 5-dim for HDBSCAN input
- UMAP → 2-dim for visualization only (separate fit)

**Clustering:**
- Primary: HDBSCAN (`min_cluster_size=30`, `min_samples=5` — both in `config.yaml`)
- Noise points (label -1) tracked separately, not forced into clusters
- Secondary: KMeans with k = number of HDBSCAN clusters (for comparison)

**Stage labeling:**
- Sample 10 reviews closest to each cluster centroid (cosine similarity in 384-dim embedding space)
- Prompt Ollama (`llama3.1:8b`) to assign: `stage_name`, `ttm_stage`, `description`, `risk_level`
- Output: `outputs/tables/cluster_stages.json`

---

## Three-Approach Classification

All three output `risk_level` ∈ {`HIGH`, `MODERATE`, `LOW`} for direct comparison.

**Approach 1 — Keyword Baseline (`baseline.py`):**
- HIGH: relapse/overdose/despair regex patterns
- LOW: recovery/sobriety/stability regex patterns
- MODERATE: everything else
- All patterns in `config.yaml`

**Approach 2 — Embedding + Cluster (`clustering.py`):**
- Each cluster's `risk_level` derived from its LLM stage label
- New post classified by nearest-centroid lookup
- No LLM call at inference time

**Approach 3 — Direct LLM (`classify.py`):**
- Per-review prompt → `stage_label` + `risk_level` + one-sentence rationale
- Batched via Ollama; `BATCH_SIZE` in `config.yaml`
- Rationale stored as explainability output

**Evaluation (`evaluate.py`):**
- LLM labels as pseudo-ground-truth
- Precision/recall/F1 for baseline and cluster approach vs. LLM
- Output: `outputs/tables/approach_comparison.csv`

---

## Temporal Analysis

**Stage Distribution Drift:**
- Group by `year_quarter`, normalize to % per stage per quarter
- Stacked area chart → `outputs/figures/stage_drift.png`

**Drug Effectiveness Trends:**
- Group by `drugName` × `year_quarter`, compute median rating + dominant stage
- Output: `outputs/tables/drug_trends.csv` + `outputs/figures/drug_trends.png`

**Spike Detection (hero chart):**
- Quarterly frequency of HIGH-risk reviews
- Z-score anomaly detection (rolling 4-quarter window, threshold=2.0 — in `config.yaml`)
- Flagged spikes → LLM reads top-20 HIGH-risk reviews from that quarter (sorted by `usefulCount` descending) → 3-sentence narrative per spike
- Narratives: `outputs/summaries/spike_narratives.json`
- Hero chart: line plot + shaded spike periods + narrative annotations → `outputs/figures/spike_detection.png`

---

## Explainability

**Cluster summaries (`explain.py`):**
- One LLM-generated summary per discovered stage
- System prompt frames LLM as "public health data analyst summarizing population-level trends"
- Format: stage name, % of corpus, behavioral description, public health implication
- Output: `outputs/summaries/cluster_summaries.md`

**Spike narratives:**
- Generated inline in `temporal.py` during spike detection
- Output: `outputs/summaries/spike_narratives.json`

---

## Streamlit Dashboard (`app.py`)

Required deliverable. Four views:

1. **Overview** — corpus stats, filter breakdown, cluster count
2. **Recovery Stages** — 2D UMAP scatter colored by stage + cluster summary panel
3. **Temporal Analysis** — interactive spike detection chart + stage drift area chart
4. **Drug Comparison** — filterable drug trends table + chart

---

## Output Inventory

| File | Purpose |
|------|---------|
| `outputs/figures/spike_detection.png` | Report Fig 1, video, dashboard |
| `outputs/figures/stage_drift.png` | Report Fig 2, dashboard |
| `outputs/figures/drug_trends.png` | Report Fig 3, dashboard |
| `outputs/figures/umap_clusters.png` | Report Fig 4, dashboard |
| `outputs/tables/approach_comparison.csv` | Report Table 1 |
| `outputs/tables/cluster_stages.json` | Stage labels + TTM mapping |
| `outputs/tables/drug_trends.csv` | Drug effectiveness data |
| `outputs/summaries/cluster_summaries.md` | Report appendix, dashboard |
| `outputs/summaries/spike_narratives.json` | Chart annotations, dashboard |

---

## Ethical Constraints

- No PII. Population-level insights only.
- All LLM prompts explicitly frame outputs as population-level, not individual assessment.
- No surveillance framing. Public health awareness only.
- Dataset is anonymized patient reviews — no re-identification attempted.

---

## Config-Driven Design

All tunable parameters in `config.yaml`:
- Addiction condition/drug filter lists
- Embedding model name
- UMAP n_components, n_neighbors, min_dist
- HDBSCAN min_cluster_size, min_samples
- KMeans n_clusters (auto-set or manual override)
- LLM model name, batch size, temperature
- Spike detection window size and z-score threshold
- Random seeds (all modules)
- File paths

---

## Session Plan

| Session | Focus | Modules |
|---------|-------|---------|
| 1 | Data ingestion + EDA | `ingest.py`, `preprocess.py` |
| 2 | Embedding pipeline | `embeddings.py`, `clustering.py` |
| 3 | Risk classification | `baseline.py`, `classify.py`, `evaluate.py` |
| 4 | Temporal analysis | `temporal.py` |
| 5 | Explainability + dashboard | `explain.py`, `app.py` |
| 6 | Polish | README, `reproduce.sh`, figures, cleanup |
