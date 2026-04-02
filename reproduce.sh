#!/usr/bin/env bash
# reproduce.sh — run the full High Alert pipeline from scratch
set -euo pipefail

echo "=== High Alert: Recovery Trajectory Staging Pipeline ==="
echo ""

echo "[1/6] Ingesting and filtering data..."
.venv/bin/python -m src.ingest

echo "[2/6] Preprocessing text..."
.venv/bin/python -m src.preprocess

echo "[3/6] Generating embeddings..."
.venv/bin/python -m src.embeddings

echo "[4/6] Clustering and labeling recovery stages..."
.venv/bin/python -m src.clustering

echo "[5/6] Running risk classification (baseline + LLM)..."
.venv/bin/python -m src.baseline
.venv/bin/python -m src.classify
.venv/bin/python -m src.evaluate

echo "[6/6] Temporal analysis and explainability..."
.venv/bin/python -m src.temporal
.venv/bin/python -m src.explain

echo ""
echo "=== Pipeline complete. ==="
echo "Outputs saved to outputs/"
echo "Launch dashboard: .venv/bin/streamlit run app.py"
