"""
insights.py — Business Insights & KPI Generator
AI Data Analyst Assistant

Auto-generates KPIs, top/bottom performers, trends, and AI recommendations.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ──────────────────────────────────────────
# KPI Extraction
# ──────────────────────────────────────────
def extract_kpis(df: pd.DataFrame) -> list[dict]:
    """
    Automatically extract key business KPIs from the DataFrame.
    Returns a list of KPI dicts: {label, value, delta, format, icon}.
    """
    kpis = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Map common column name patterns to KPI labels
    kpi_patterns = {
        "sales":   ("💰", "Total Sales", "currency"),
        "revenue": ("📈", "Total Revenue", "currency"),
        "profit":  ("💹", "Total Profit", "currency"),
        "orders":  ("📦", "Total Orders", "integer"),
        "order":   ("📦", "Total Orders", "integer"),
        "quantity": ("🛒", "Total Quantity", "integer"),
        "qty":     ("🛒", "Total Quantity", "integer"),
        "discount": ("🏷️", "Avg Discount %", "percent"),
        "margin":  ("📊", "Profit Margin %", "percent"),
        "cost":    ("💸", "Total Cost", "currency"),
        "amount":  ("💵", "Total Amount", "currency"),
    }

    found_labels = set()

    for col in numeric_cols:
        col_lower = col.lower()
        for pattern, (icon, label, fmt) in kpi_patterns.items():
            if pattern in col_lower and label not in found_labels:
                val = df[col].sum() if fmt in ("currency", "integer") else df[col].mean()
                found_labels.add(label)
                kpis.append({"label": label, "value": val, "icon": icon, "format": fmt, "col": col})
                break

    # Fallback: Total and averages for first 4 numeric cols
    if len(kpis) < 2:
        for col in numeric_cols[:4]:
            label = f"Total {col}"
            if label not in found_labels:
                kpis.append({
                    "label": label,
                    "value": df[col].sum(),
                    "icon": "📊",
                    "format": "number",
                    "col": col,
                })
                found_labels.add(label)

    # Always add row count and column count
    kpis.insert(0, {"label": "Total Records", "value": len(df), "icon": "🗂️", "format": "integer", "col": None})

    return kpis[:8]  # Cap at 8 KPIs


def format_kpi_value(value: float, fmt: str) -> str:
    """Format a KPI value for display."""
    if fmt == "currency":
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"${value / 1_000:.1f}K"
        return f"${value:,.2f}"
    elif fmt == "percent":
        return f"{value:.1f}%"
    elif fmt == "integer":
        return f"{int(value):,}"
    else:
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:,.2f}"


# ──────────────────────────────────────────
# Top / Bottom Performers
# ──────────────────────────────────────────
def get_top_performers(df: pd.DataFrame, group_col: str, value_col: str, n: int = 10) -> pd.DataFrame:
    """Return top N performers for a given grouping."""
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(group_col)[value_col]
        .sum()
        .reset_index()
        .sort_values(value_col, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def get_bottom_performers(df: pd.DataFrame, group_col: str, value_col: str, n: int = 10) -> pd.DataFrame:
    """Return bottom N performers for a given grouping."""
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(group_col)[value_col]
        .sum()
        .reset_index()
        .sort_values(value_col, ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────
# Trend Analysis
# ──────────────────────────────────────────
def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a date/time column in the DataFrame."""
    # Check existing datetime columns
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if datetime_cols:
        return datetime_cols[0]

    # Attempt to parse object columns
    date_keywords = ["date", "time", "month", "year", "period", "day"]
    for col in df.select_dtypes(include="object").columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                pd.to_datetime(df[col].head(20), errors="raise")
                return col
            except Exception:
                pass
    return None


def get_monthly_trend(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Aggregate values by month."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["Month"] = df[date_col].dt.to_period("M").astype(str)
    return (
        df.groupby("Month")[value_col]
        .sum()
        .reset_index()
        .sort_values("Month")
    )


def get_quarterly_trend(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Aggregate values by quarter."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["Quarter"] = df[date_col].dt.to_period("Q").astype(str)
    return (
        df.groupby("Quarter")[value_col]
        .sum()
        .reset_index()
        .sort_values("Quarter")
    )


def get_yearly_trend(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Aggregate values by year."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["Year"] = df[date_col].dt.year.astype(str)
    return (
        df.groupby("Year")[value_col]
        .sum()
        .reset_index()
        .sort_values("Year")
    )


def calculate_growth(trend_df: pd.DataFrame, value_col: str) -> float:
    """Calculate period-over-period growth percentage."""
    if len(trend_df) < 2:
        return 0.0
    first_val = trend_df.iloc[0][value_col]
    last_val = trend_df.iloc[-1][value_col]
    if first_val == 0:
        return 0.0
    return round(((last_val - first_val) / abs(first_val)) * 100, 2)


# ──────────────────────────────────────────
# Auto Business Insights
# ──────────────────────────────────────────
def generate_auto_insights(df: pd.DataFrame) -> list[dict]:
    """
    Generate structured business insight cards from the data.
    Each insight is: {title, value, subtitle, icon, color}
    """
    insights = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if not numeric_cols:
        return insights

    primary_metric = numeric_cols[0]

    # Top Category
    if cat_cols:
        top_cat_col = cat_cols[0]
        grouped = df.groupby(top_cat_col)[primary_metric].sum().sort_values(ascending=False)
        if not grouped.empty:
            insights.append({
                "title": f"Best {top_cat_col}",
                "value": str(grouped.index[0]),
                "subtitle": f"${grouped.iloc[0]:,.2f} {primary_metric}",
                "icon": "🏆", "color": "#667eea",
            })
            insights.append({
                "title": f"Worst {top_cat_col}",
                "value": str(grouped.index[-1]),
                "subtitle": f"${grouped.iloc[-1]:,.2f} {primary_metric}",
                "icon": "⚠️", "color": "#f5576c",
            })

    # Highest and Lowest values
    insights.append({
        "title": f"Highest {primary_metric}",
        "value": f"${df[primary_metric].max():,.2f}",
        "subtitle": "Maximum recorded value",
        "icon": "📈", "color": "#00f2fe",
    })
    insights.append({
        "title": f"Lowest {primary_metric}",
        "value": f"${df[primary_metric].min():,.2f}",
        "subtitle": "Minimum recorded value",
        "icon": "📉", "color": "#f093fb",
    })

    # Average
    insights.append({
        "title": f"Avg {primary_metric}",
        "value": f"${df[primary_metric].mean():,.2f}",
        "subtitle": f"Std: {df[primary_metric].std():,.2f}",
        "icon": "📊", "color": "#4facfe",
    })

    # Second category performer
    if len(cat_cols) > 1:
        second_cat = cat_cols[1]
        grouped2 = df.groupby(second_cat)[primary_metric].sum().sort_values(ascending=False)
        if not grouped2.empty:
            insights.append({
                "title": f"Top {second_cat}",
                "value": str(grouped2.index[0]),
                "subtitle": f"${grouped2.iloc[0]:,.2f} {primary_metric}",
                "icon": "🌟", "color": "#43e97b",
            })

    # Trend with date column
    date_col = detect_date_column(df)
    if date_col and primary_metric in df.columns:
        try:
            trend = get_yearly_trend(df, date_col, primary_metric)
            if len(trend) >= 2:
                growth = calculate_growth(trend, primary_metric)
                color = "#43e97b" if growth >= 0 else "#f5576c"
                icon = "📈" if growth >= 0 else "📉"
                insights.append({
                    "title": "Overall Growth",
                    "value": f"{'+' if growth >= 0 else ''}{growth}%",
                    "subtitle": f"First to last period",
                    "icon": icon, "color": color,
                })
        except Exception:
            pass

    return insights
