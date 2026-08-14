"""
EduPro UI Helper & Visualization Utility Functions for Streamlit
Provides custom CSS themes, Plotly visualizations, metric cards, and badge UI components.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as gg
import pandas as pd

def apply_custom_css():
    """
    Inject modern dark glassmorphism CSS styling into Streamlit app.
    """
    st.markdown("""
    <style>
    /* Main App Background & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Header Container Styling */
    .header-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(129, 140, 248, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(192, 132, 252, 0.5);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
    }
    
    /* Segment Badge Styling */
    .segment-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
    }
    .badge-power { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid #ec4899; }
    .badge-specialist { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #a855f7; }
    .badge-developing { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .badge-explorer { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

    /* Course Card Styling */
    .course-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.25s ease;
    }
    .course-card:hover {
        border-color: #818cf8;
        box-shadow: 0 4px 20px rgba(129, 140, 248, 0.25);
    }
    .course-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .course-meta {
        font-size: 0.88rem;
        color: #cbd5e1;
        display: flex;
        gap: 16px;
        margin-bottom: 12px;
    }
    .teacher-box {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Recommendation Reason Badges */
    .reason-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
        border-radius: 4px;
        padding: 8px 12px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .reason-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: #34d399;
        margin-bottom: 4px;
    }
    .reason-item {
        font-size: 0.83rem;
        color: #cbd5e1;
        margin: 2px 0;
    }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


def plot_elbow_curve(eval_df, selected_k=4):
    """
    Plot Elbow curve using Plotly.
    """
    fig = px.line(
        eval_df,
        x="K",
        y="Inertia",
        markers=True,
        title="<b>Elbow Method (Inertia vs. Number of Clusters K)</b>",
        labels={"Inertia": "Inertia (Within-Cluster Sum of Squares)", "K": "Number of Clusters (K)"}
    )
    fig.update_traces(line_color="#818cf8", line_width=3, marker=dict(size=10, color="#c084fc"))
    
    # Highlight selected K
    sel_row = eval_df[eval_df["K"] == selected_k]
    if not sel_row.empty:
        fig.add_scatter(
            x=[selected_k],
            y=[sel_row["Inertia"].iloc[0]],
            mode="markers+text",
            marker=dict(size=16, color="#ef4444", symbol="star"),
            text=[f"Selected K={selected_k}"],
            textposition="top center",
            name=f"Selected K={selected_k}"
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.5)",
        font=dict(family="Plus Jakarta Sans", color="#f8fafc"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_silhouette_curve(eval_df, selected_k=4):
    """
    Plot Silhouette score curve using Plotly.
    """
    fig = px.bar(
        eval_df,
        x="K",
        y="SilhouetteScore",
        text_auto=".3f",
        title="<b>Silhouette Scores (Higher score indicates better cluster separation)</b>",
        labels={"SilhouetteScore": "Silhouette Score", "K": "Number of Clusters (K)"}
    )
    
    colors = ["#818cf8" if k != selected_k else "#ec4899" for k in eval_df["K"]]
    fig.update_traces(marker_color=colors)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.5)",
        font=dict(family="Plus Jakarta Sans", color="#f8fafc"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_pca_clusters(learner_df, segment_map):
    """
    2D PCA Scatter plot showing cluster separation in space.
    """
    df_plot = learner_df.copy()
    df_plot["Segment"] = df_plot["Cluster"].map(segment_map)

    fig = px.scatter(
        df_plot,
        x="PCA1",
        y="PCA2",
        color="Segment",
        hover_data=["UserID", "UserName", "TotalCourses", "TotalSpending", "PreferredCategory"],
        title="<b>2D Visual Mapping of Learner Clusters (PCA Projection)</b>",
        color_discrete_sequence=["#38bdf8", "#c084fc", "#34d399", "#f472b6", "#fbbf24", "#a3e635"]
    )

    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.3)")))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.5)",
        font=dict(family="Plus Jakarta Sans", color="#f8fafc"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(title="Learner Segment", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_cluster_feature_comparison(profiles_df, feature="AvgSpending"):
    """
    Plot bar comparison of a specific feature across segments.
    """
    fig = px.bar(
        profiles_df,
        x="SegmentName",
        y=feature,
        color="SegmentName",
        text_auto=True,
        title=f"<b>Segment Comparison: {feature}</b>",
        color_discrete_sequence=["#818cf8", "#c084fc", "#34d399", "#f472b6", "#38bdf8"]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.5)",
        font=dict(family="Plus Jakarta Sans", color="#f8fafc"),
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def get_segment_badge_html(segment_name):
    """
    Generate colored badge HTML element for a segment name.
    """
    s = str(segment_name).lower()
    if "power" in s or "high-value" in s:
        cls = "badge-power"
    elif "specialist" in s or "advanced" in s:
        cls = "badge-specialist"
    elif "developing" in s or "intermediate" in s:
        cls = "badge-developing"
    else:
        cls = "badge-explorer"
    
    return f'<span class="segment-badge {cls}">{segment_name}</span>'
