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
    risk_counts = df["baseline_risk"].value_counts().reset_index()
    risk_counts.columns = ["baseline_risk", "count"]
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = {"HIGH": "#d62728", "MODERATE": "#ff7f0e", "LOW": "#2ca02c"}
    ax.bar(
        risk_counts["baseline_risk"],
        risk_counts["count"],
        color=[colors.get(str(r), "#1f77b4") for r in risk_counts["baseline_risk"]],
    )
    ax.set_ylabel("Review Count")
    st.pyplot(fig)
    plt.close()

    st.subheader("Approach Comparison")
    comp_path = "outputs/tables/approach_comparison.csv"
    if os.path.exists(comp_path):
        st.dataframe(pd.read_csv(comp_path), use_container_width=True)
    else:
        st.info("Run `.venv/bin/python -m src.evaluate` to generate approach comparison.")

# --- Tab 2: Recovery Stages ---
with tab2:
    st.title("Recovery Stage Discovery")
    st.markdown("Stages discovered via HDBSCAN clustering, labeled by LLM.")

    if "umap_x" in df.columns:
        st.subheader("2D UMAP Projection by Recovery Stage")
        stages = df["stage_name"].unique()
        palette = sns.color_palette("tab10", len(stages))
        color_map = {
            stage: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for stage, (r, g, b) in zip(stages, palette)
        }

        fig, ax = plt.subplots(figsize=(10, 7))
        for stage in stages:
            mask = df["stage_name"] == stage
            ax.scatter(
                df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                c=[color_map[stage]], label=stage, s=5, alpha=0.6,
            )
        ax.legend(fontsize=8, markerscale=3)
        ax.set_title("Recovery Stage Clusters (UMAP 2D)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run `.venv/bin/python -m src.clustering` to generate UMAP coordinates.")

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
        st.info("Run `.venv/bin/python -m src.temporal` to generate charts.")

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
            all_quarters = sorted(drug_trends["year_quarter"].unique())
            quarter_to_idx = {q: i for i, q in enumerate(all_quarters)}
            fig, ax = plt.subplots(figsize=(12, 5))
            for drug in selected_drugs:
                d = filtered[filtered["drugName"] == drug].sort_values(by="year_quarter")  # type: ignore[call-overload]
                x = [quarter_to_idx[q] for q in d["year_quarter"]]
                ax.plot(x, d["median_rating"], label=drug, linewidth=1.5, marker=".")
            step = max(1, len(all_quarters) // 10)
            ax.set_xticks(range(0, len(all_quarters), step))
            ax.set_xticklabels(
                [all_quarters[i] for i in range(0, len(all_quarters), step)],
                rotation=45, ha="right", fontsize=8,
            )
            ax.set_xlabel("Quarter")
            ax.set_ylabel("Median Rating (1–10)")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()

            st.subheader("Raw Trends Data")
            st.dataframe(filtered.sort_values(by=["drugName", "year_quarter"]), use_container_width=True)  # type: ignore[call-overload]
    else:
        st.info("Run `.venv/bin/python -m src.temporal` to generate drug trends data.")
