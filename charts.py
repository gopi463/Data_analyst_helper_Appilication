"""
charts.py — Plotly Chart Builder
AI Data Analyst Assistant

Supports 10+ chart types with auto-suggestion and custom builder.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


# ──────────────────────────────────────────
# Theme Config
# ──────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = px.colors.qualitative.Vivid
GRADIENT_COLORS = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe"]


def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=18, color="#e6edf3")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3", family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    )


# ──────────────────────────────────────────
# Individual Chart Builders
# ──────────────────────────────────────────
def bar_chart(df: pd.DataFrame, x: str, y: str, color: str = None, title: str = "") -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title,
                 color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE)
    fig.update_layout(**_base_layout(title))
    fig.update_traces(marker_line_width=0)
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, color: str = None, title: str = "") -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, title=title,
                  color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE)
    fig.update_layout(**_base_layout(title))
    fig.update_traces(line_width=2.5)
    return fig


def pie_chart(df: pd.DataFrame, names: str, values: str, title: str = "") -> go.Figure:
    fig = px.pie(df, names=names, values=values, title=title,
                 color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE,
                 hole=0.35)
    fig.update_layout(**_base_layout(title))
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def histogram_chart(df: pd.DataFrame, x: str, nbins: int = 30, color: str = None, title: str = "") -> go.Figure:
    fig = px.histogram(df, x=x, nbins=nbins, color=color, title=title,
                       color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE)
    fig.update_layout(**_base_layout(title))
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.2)")
    return fig


def scatter_chart(df: pd.DataFrame, x: str, y: str, color: str = None, size: str = None, title: str = "") -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title,
                     color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE,
                     opacity=0.8)
    fig.update_layout(**_base_layout(title))
    return fig


def area_chart(df: pd.DataFrame, x: str, y: str, color: str = None, title: str = "") -> go.Figure:
    fig = px.area(df, x=x, y=y, color=color, title=title,
                  color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE)
    fig.update_layout(**_base_layout(title))
    return fig


def box_chart(df: pd.DataFrame, x: str = None, y: str = None, color: str = None, title: str = "") -> go.Figure:
    fig = px.box(df, x=x, y=y, color=color, title=title,
                 color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE)
    fig.update_layout(**_base_layout(title))
    return fig


def heatmap_chart(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(**_base_layout(title))
    return fig


def treemap_chart(df: pd.DataFrame, path: list, values: str, color: str = None, title: str = "") -> go.Figure:
    fig = px.treemap(df, path=path, values=values, color=color, title=title,
                     color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE)
    fig.update_layout(**_base_layout(title))
    fig.update_traces(textinfo="label+percent entry")
    return fig


def sunburst_chart(df: pd.DataFrame, path: list, values: str, title: str = "") -> go.Figure:
    fig = px.sunburst(df, path=path, values=values, title=title,
                      color_discrete_sequence=GRADIENT_COLORS, template=PLOTLY_TEMPLATE)
    fig.update_layout(**_base_layout(title))
    return fig


# ──────────────────────────────────────────
# Auto Chart Builder
# ──────────────────────────────────────────
def auto_suggest_charts(df: pd.DataFrame) -> list[dict]:
    """
    Auto-suggest relevant charts based on data types.
    Returns a list of chart specs: {type, params, title}
    """
    suggestions = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Distribution histograms for numeric cols
    for col in numeric_cols[:3]:
        suggestions.append({
            "type": "histogram",
            "params": {"x": col},
            "title": f"Distribution of {col}",
        })

    # Bar chart: categorical vs numeric
    if cat_cols and numeric_cols:
        suggestions.append({
            "type": "bar",
            "params": {"x": cat_cols[0], "y": numeric_cols[0]},
            "title": f"{numeric_cols[0]} by {cat_cols[0]}",
        })

    # Pie chart for first categorical
    if cat_cols and numeric_cols:
        suggestions.append({
            "type": "pie",
            "params": {"names": cat_cols[0], "values": numeric_cols[0]},
            "title": f"{numeric_cols[0]} Distribution by {cat_cols[0]}",
        })

    # Scatter for 2 numeric
    if len(numeric_cols) >= 2:
        suggestions.append({
            "type": "scatter",
            "params": {"x": numeric_cols[0], "y": numeric_cols[1]},
            "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
        })

    # Heatmap
    if len(numeric_cols) >= 3:
        suggestions.append({
            "type": "heatmap",
            "params": {},
            "title": "Correlation Heatmap",
        })

    return suggestions


def build_chart(df: pd.DataFrame, chart_type: str, params: dict, title: str = "") -> Optional[go.Figure]:
    """
    Build a Plotly chart from a type string and params dict with smart key normalization.

    chart_type: 'bar' | 'line' | 'pie' | 'histogram' | 'scatter' |
                'area' | 'box' | 'heatmap' | 'treemap' | 'sunburst'
    """
    try:
        # Create a copy of params to avoid mutating input dict
        p = dict(params)
        ct = chart_type.lower().replace(" chart", "").replace(" ", "_")

        # ── Parameter Key Normalization ───────────────────────────────────────
        if ct == "pie":
            if "names" not in p and "x" in p:
                p["names"] = p.pop("x")
            if "values" not in p and "y" in p:
                p["values"] = p.pop("y")
        elif ct in ("treemap", "sunburst"):
            if "path" not in p and "x" in p:
                p["path"] = [p.pop("x")]
            if "values" not in p and "y" in p:
                p["values"] = p.pop("y")
            # Ensure path is a list
            if "path" in p and isinstance(p["path"], str):
                p["path"] = [p["path"]]
        elif ct in ("histogram", "hist"):
            # Histogram doesn't strictly need a 'y' axis (it defaults to count)
            if "x" not in p and "y" in p:
                p["x"] = p.pop("y")
            # Remove y if present to prevent 2D histograms unless explicitly wanted
            p.pop("y", None)
        elif ct in ("bar", "line", "scatter", "area", "box"):
            # If color is 'None' as string, remove it
            if p.get("color") == "None":
                p.pop("color")

        # ── Chart Dispatch ────────────────────────────────────────────────────
        if ct == "bar":
            return bar_chart(df, title=title, **p)
        elif ct == "line":
            return line_chart(df, title=title, **p)
        elif ct == "pie":
            return pie_chart(df, title=title, **p)
        elif ct in ("histogram", "hist"):
            return histogram_chart(df, title=title, **p)
        elif ct == "scatter":
            return scatter_chart(df, title=title, **p)
        elif ct == "area":
            return area_chart(df, title=title, **p)
        elif ct == "box":
            return box_chart(df, title=title, **p)
        elif ct == "heatmap":
            return heatmap_chart(df, title=title)
        elif ct == "treemap":
            return treemap_chart(df, title=title, **p)
        elif ct == "sunburst":
            return sunburst_chart(df, title=title, **p)
        else:
            return None
    except Exception:
        return None


# ──────────────────────────────────────────
# KPI Sparkline
# ──────────────────────────────────────────
def sparkline(values: list, color: str = "#667eea") -> go.Figure:
    """Mini sparkline for KPI cards."""
    fig = go.Figure(go.Scatter(
        y=values, mode="lines", line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba(102,126,234,0.15)",
    ))
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
