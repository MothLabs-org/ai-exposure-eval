"""
AI Automation Exposure Dashboard (Phase 3)
Visualises LLM evaluation results for multiple models across 114 digital occupations.
Includes Sycophancy Analysis.
Run:  streamlit run dashboard.py
"""

import json
import glob
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Exposure & Sycophancy Dashboard",
    page_icon="🤖",
    layout="wide",
)

ALL_DIMENSIONS = ["Correctness", "Completeness", "Best Practices", "Domain Accuracy", "Clarity", "Autonomy"]
CORE_DIMENSIONS = ["Correctness", "Completeness", "Best Practices", "Domain Accuracy", "Clarity"]

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_all_results():
    raw_data = {}
    rows = []

    # Use script location as base so it works regardless of CWD
    base = os.path.dirname(os.path.abspath(__file__))

    # Map of directory name → friendly model label
    DIR_LABELS = {
        "results":         "Qwen 35B",
        "results_1.5b_v2": "Qwen 1.5B",
        "results_gemma3_4b": "Gemma 3 4B",
        "results_7b":      "Qwen 7B",
    }

    for dir_name, model_label in DIR_LABELS.items():
        pattern = os.path.join(base, dir_name, "*_results_auto.json")
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, encoding="utf-8") as f:
                    d = json.load(f)
                m = d.get("meta", {})
                score = m.get("exposure_score", {})
                overall = score.get("overall", 0) if isinstance(score, dict) else 0
                job_title = m.get("job_title", "Unknown")

                if not overall or overall <= 0:
                    continue

                raw_data[(model_label, job_title)] = d

                dims = score.get("dimensions", {}) if isinstance(score, dict) else {}
                row = {
                    "Model": model_label,
                    "Job Title": job_title,
                    "Overall Score": overall,
                    "Evaluated Tasks": m.get("evaluated_tasks", 0),
                    "Total O*NET Tasks": m.get("total_onet_tasks", 0),
                }
                for dim in ALL_DIMENSIONS:
                    row[dim] = dims.get(dim, None)
                rows.append(row)
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Model", "Overall Score"], ascending=[True, False]).reset_index(drop=True)
    return raw_data, df


raw_data, all_df = load_all_results()

if all_df.empty:
    st.error("No results found. Run the pipeline first to generate data.")
    st.stop()

# ── Helper: get available dimensions for a model (skip all-null columns) ───────
def get_available_dims(df_subset):
    available = []
    for dim in ALL_DIMENSIONS:
        if dim in df_subset.columns and df_subset[dim].notna().any():
            available.append(dim)
    return available

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("Navigation")

available_models = sorted(all_df["Model"].unique().tolist())
selected_model = st.sidebar.selectbox("🎯 Context: Select Model", available_models)

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Job Detail", "Dimension Analysis", "Sycophancy Analysis"],
    index=0,
)
st.sidebar.markdown("---")

# Filtered DF for single-model pages
model_df = all_df[all_df["Model"] == selected_model].copy()
model_df.insert(0, "Rank", range(1, len(model_df) + 1))
model_dims = get_available_dims(model_df)

st.sidebar.metric(f"Jobs Evaluated ({selected_model})", len(model_df))
st.sidebar.metric(f"Avg Exposure ({selected_model})", f"{model_df['Overall Score'].mean():.2f} / 5")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title(f"🤖 Exposure Overview: {selected_model}")
    st.markdown("Quantifying how well this LLM performs across digital occupations.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jobs Evaluated", len(model_df))
    col2.metric("Avg Exposure Score", f"{model_df['Overall Score'].mean():.2f} / 5")
    col3.metric("Score Range", f"{model_df['Overall Score'].min():.2f} – {model_df['Overall Score'].max():.2f}")

    st.markdown("---")
    st.subheader("Occupation Exposure Score Ranking")

    top_n = st.slider("Show top / bottom N jobs", min_value=5, max_value=max(10, len(model_df)), value=min(30, len(model_df)), step=5)
    view = st.radio("View", ["Top (most exposed)", "Bottom (least exposed)", "All"], horizontal=True)

    if view == "Top (most exposed)":
        plot_df = model_df.head(top_n).copy()
    elif view == "Bottom (least exposed)":
        plot_df = model_df.tail(top_n).sort_values("Overall Score").copy()
    else:
        plot_df = model_df.copy()

    color_scale = px.colors.diverging.RdYlGn
    fig_bar = px.bar(
        plot_df, x="Overall Score", y="Job Title", orientation="h",
        color="Overall Score", color_continuous_scale=color_scale,
        range_color=[2.5, 5.0], text="Overall Score", height=max(500, len(plot_df) * 22),
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, xaxis_range=[0, 5.5])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Dimension Score Heatmap (Top 30) — {len(model_dims)} dimensions")
    if "Autonomy" not in model_dims:
        st.caption("ℹ️ Autonomy dimension not available for this model — showing 5 core dimensions.")
    heat_df = model_df.head(30).set_index("Job Title")[model_dims].fillna(0)
    fig_heat = px.imshow(heat_df, color_continuous_scale="RdYlGn", zmin=1, zmax=5, aspect="auto", height=650)
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — JOB DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Job Detail":
    st.title(f"Job Detail — {selected_model}")

    job_list = model_df["Job Title"].tolist()
    selected_job = st.selectbox("Select an occupation", options=job_list)

    if selected_job:
        d = raw_data.get((selected_model, selected_job))
        if not d:
            st.error("Data not found.")
        else:
            m = d["meta"]
            score = m.get("exposure_score", {})

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Exposure Score", f"{score.get('overall', 0):.2f} / 5")
            col2.metric("Evaluated Tasks", m.get("evaluated_tasks", "—"))
            col3.metric("O*NET Code", m.get("onet_code", "—"))

            dims = score.get("dimensions", {}) if isinstance(score, dict) else {}
            if dims:
                st.markdown("---")
                col_bar, col_radar = st.columns([1, 1])

                with col_bar:
                    st.subheader("Dimension Averages")
                    dim_df = pd.DataFrame({"Dimension": list(dims.keys()), "Score": list(dims.values())})
                    fig_dim = px.bar(dim_df, x="Dimension", y="Score", color="Score", color_continuous_scale="RdYlGn", range_color=[1, 5], range_y=[0, 5.5], text="Score", height=360)
                    fig_dim.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig_dim.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_dim, use_container_width=True)

                with col_radar:
                    st.subheader("Radar Profile")
                    dim_keys = [dim for dim in ALL_DIMENSIONS if dim in dims]
                    values = [dims[dim] for dim in dim_keys]
                    fig_r = go.Figure(go.Scatterpolar(r=values+[values[0]], theta=dim_keys+[dim_keys[0]], fill="toself", fillcolor="rgba(76,120,168,0.3)", line=dict(color="#4C78A8", width=2)))
                    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False, height=360)
                    st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIMENSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dimension Analysis":
    st.title(f"Dimension Analysis — {selected_model}")
    st.markdown("---")

    st.subheader(f"Score Distribution per Dimension (Box Plot) — {len(model_dims)} dimensions")
    if "Autonomy" not in model_dims:
        st.caption("ℹ️ Autonomy dimension not available for this model.")
    box_rows = [{"Dimension": dim, "Score": val} for dim in model_dims for val in model_df[dim].dropna()]
    box_df = pd.DataFrame(box_rows)
    fig_box = px.box(box_df, x="Dimension", y="Score", color="Dimension", points="all", range_y=[0, 5.5], height=420)
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AUTONOMY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Autonomy Analysis":
    st.title("🤖 Autonomy Analysis")
    st.markdown("""
    **Autonomy** is a 6th evaluation dimension introduced to go beyond task quality and ask:
    *Can an AI's output be used as-is, or does it still require significant human rework?*

    This dimension bridges the gap between Anthropic's concepts of **automation** (AI replaces the work)
    vs. **augmentation** (AI assists the work). A high Autonomy score means the output is deployment-ready;
    a low score means it still requires expert review or heavy editing.

    > ⚠️ **Note:** Autonomy was introduced in the 6-dimension rubric and is only scored for models
    > evaluated under the updated pipeline (e.g., Qwen 1.5B v2, Gemma 3 4B). The base Qwen 35B
    > run uses the original 5-dimension rubric and does not include this dimension.
    """)
    st.markdown("---")

    # Filter to only models that have Autonomy data
    auto_df = all_df[all_df["Autonomy"].notna()].copy()

    if auto_df.empty:
        st.warning("""
        No Autonomy data is available in the currently deployed results.

        Autonomy scores require the 6-dimension rubric (introduced in newer pipeline runs).
        The base deployment only includes the Qwen 35B run which uses the 5-dimension rubric.
        """)
        st.info("To see Autonomy data, re-run the pipeline with `results_1.5b_v2` or `results_gemma3_4b` committed to the repository.")
    else:
        auto_models = auto_df["Model"].unique().tolist()
        st.success(f"Autonomy data available for: **{', '.join(auto_models)}**")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Autonomy Score by Model")
            avg_auto = auto_df.groupby("Model")["Autonomy"].mean().reset_index().sort_values("Autonomy", ascending=False)
            fig_avg = px.bar(avg_auto, x="Model", y="Autonomy", color="Model", text="Autonomy",
                             range_y=[0, 5.5], height=380)
            fig_avg.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_avg.update_layout(coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(fig_avg, use_container_width=True)

        with col2:
            st.subheader("Autonomy Score Distribution (Box Plot)")
            fig_box_a = px.box(auto_df, x="Model", y="Autonomy", color="Model",
                               points="all", range_y=[0, 5.5], height=380)
            fig_box_a.update_layout(showlegend=False)
            st.plotly_chart(fig_box_a, use_container_width=True)

        st.markdown("---")
        st.subheader("Autonomy vs. Overall Exposure Score")
        st.caption("Does a high exposure score always mean a high autonomy? High exposure + low autonomy = AI can do the task but still needs human oversight.")
        fig_scatter_a = px.scatter(
            auto_df, x="Overall Score", y="Autonomy", color="Model",
            hover_name="Job Title", opacity=0.7, height=500,
            trendline="ols"
        )
        fig_scatter_a.update_layout(xaxis_title="Overall Exposure Score", yaxis_title="Autonomy Score")
        st.plotly_chart(fig_scatter_a, use_container_width=True)

        st.markdown("---")
        st.subheader("Top 20 Occupations by Autonomy Score")
        top_auto = auto_df.sort_values("Autonomy", ascending=False).head(20)
        fig_top = px.bar(
            top_auto, x="Autonomy", y="Job Title", orientation="h", color="Model",
            range_x=[0, 5.5], text="Autonomy", height=580
        )
        fig_top.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SYCOPHANCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sycophancy Analysis":
    st.title("Sycophancy Analysis (The 'Fluent but Flawed' Gap)")
    st.markdown("""
    **Sycophancy** in LLMs occurs when models prioritize user approval and fluent flattery over objective accuracy. 
    Here, we measure Sycophancy via the **Fluent-Flawed Gap**: 
    `Fluent-Flawed Gap = Clarity - Domain Accuracy`
    
    A high positive gap means the model sounds incredibly convincing and clear (Clarity), but hallucinates or gets the core domain facts wrong (Domain Accuracy).
    """)
    st.markdown("---")

    syc_df = all_df.dropna(subset=["Clarity", "Domain Accuracy"]).copy()
    syc_df["Fluent-Flawed Gap"] = syc_df["Clarity"] - syc_df["Domain Accuracy"]

    st.subheader("Average Sycophancy Gap by Model")
    gap_summary = syc_df.groupby("Model")["Fluent-Flawed Gap"].mean().reset_index().sort_values("Fluent-Flawed Gap", ascending=False)

    fig_gap = px.bar(gap_summary, x="Model", y="Fluent-Flawed Gap", color="Fluent-Flawed Gap", color_continuous_scale="Reds", text="Fluent-Flawed Gap", height=400)
    fig_gap.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown("---")
    st.subheader("Clarity vs. Domain Accuracy (Scatter Plot)")
    st.caption("Models above the diagonal line are more 'Fluent' than they are 'Accurate' (Sycophancy risk).")

    fig_scatter = px.scatter(
        syc_df, x="Domain Accuracy", y="Clarity", color="Model", hover_name="Job Title", size_max=10, opacity=0.7, height=600
    )
    fig_scatter.add_shape(type="line", x0=1, y0=1, x1=5, y1=5, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig_scatter, use_container_width=True)
