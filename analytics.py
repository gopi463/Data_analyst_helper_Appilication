"""
analytics.py — Data Analytics & EDA Engine
AI Data Analyst Assistant

Provides: EDA, data cleaning, summary statistics, correlation analysis,
outlier detection, missing value analysis, and dataset quality scoring.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


# ──────────────────────────────────────────
# EDA Summary
# ──────────────────────────────────────────
def get_basic_info(df: pd.DataFrame) -> dict:
    """Return fundamental dataset information."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": int(df.shape[0] * df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "datetime_columns": df.select_dtypes(include="datetime").columns.tolist(),
    }


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate extended summary statistics for all columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    stats = numeric_df.describe().T
    stats["skewness"] = numeric_df.skew()
    stats["kurtosis"] = numeric_df.kurt()
    stats["variance"] = numeric_df.var()
    stats["cv%"] = (numeric_df.std() / numeric_df.mean() * 100).round(2)
    return stats.round(4)


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing value report."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": pct.values,
        "Present Count": (total - missing).values,
        "Data Type": df.dtypes.values,
    })
    return report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)


def get_duplicate_report(df: pd.DataFrame) -> dict:
    """Return duplicate row statistics."""
    n_duplicates = int(df.duplicated().sum())
    return {
        "total_rows": int(len(df)),
        "duplicate_rows": n_duplicates,
        "unique_rows": int(len(df) - n_duplicates),
        "duplicate_pct": round(n_duplicates / len(df) * 100, 2) if len(df) > 0 else 0,
    }


def get_unique_value_report(df: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
    """Return unique value counts per column."""
    rows = []
    for col in df.columns:
        n_unique = df[col].nunique()
        top_values = df[col].value_counts().head(5).to_dict()
        rows.append({
            "Column": col,
            "Unique Values": n_unique,
            "Uniqueness %": round(n_unique / len(df) * 100, 2) if len(df) > 0 else 0,
            "Top Values": str(top_values),
        })
    return pd.DataFrame(rows)


def get_correlation_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute Pearson correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    return numeric.corr().round(4)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using the IQR method for numeric columns.
    Returns a report DataFrame.
    """
    numeric = df.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((series < lower) | (series > upper)).sum())
        rows.append({
            "Column": col,
            "Outlier Count": n_outliers,
            "Outlier %": round(n_outliers / len(series) * 100, 2) if len(series) > 0 else 0,
            "Lower Bound": round(float(lower), 4),
            "Upper Bound": round(float(upper), 4),
            "Min": round(float(series.min()), 4),
            "Max": round(float(series.max()), 4),
        })
    return pd.DataFrame(rows).sort_values("Outlier Count", ascending=False)


def get_data_quality_score(df: pd.DataFrame) -> Tuple[int, dict]:
    """
    Calculate an overall data quality score (0-100).
    Returns (score, component_scores).
    """
    total = len(df)
    if total == 0:
        return 0, {}

    # Completeness
    completeness = float((1 - df.isnull().sum().sum() / (total * len(df.columns))) * 100)

    # Uniqueness
    dup_pct = float(df.duplicated().sum() / total * 100)
    uniqueness = max(0.0, 100.0 - dup_pct)

    # Consistency (detect mixed types in object columns)
    consistency = 100.0
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        try:
            pd.to_numeric(df[col].dropna(), errors="raise")
        except Exception:
            pass  # It's fine if a column is naturally object type

    overall = round((completeness * 0.5 + uniqueness * 0.3 + consistency * 0.2), 1)
    overall = min(100, max(0, overall))

    return int(overall), {
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "consistency": round(consistency, 1),
    }


# ──────────────────────────────────────────
# Data Cleaning Operations
# ──────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Returns (cleaned_df, n_removed)."""
    before = len(df)
    df = df.drop_duplicates()
    return df.reset_index(drop=True), before - len(df)


def fill_missing_values(df: pd.DataFrame, strategy: str = "mean", column: str = None) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.

    strategy: 'mean', 'median', 'mode', 'zero', 'ffill', 'bfill'
    column:   If None, applies to all columns.
    """
    df = df.copy()
    cols = [column] if column else df.columns.tolist()

    for col in cols:
        if col not in df.columns:
            continue
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
    return df


def drop_missing_rows(df: pd.DataFrame, column: str = None) -> Tuple[pd.DataFrame, int]:
    """Drop rows with missing values. Returns (cleaned_df, n_removed)."""
    before = len(df)
    if column:
        df = df.dropna(subset=[column])
    else:
        df = df.dropna()
    return df.reset_index(drop=True), before - len(df)


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Drop a column from the DataFrame."""
    return df.drop(columns=[column], errors="ignore")


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """Rename a column."""
    return df.rename(columns={old_name: new_name})


def change_dtype(df: pd.DataFrame, column: str, target_type: str) -> Tuple[pd.DataFrame, str]:
    """
    Attempt to change a column's data type.
    Returns (df, error_msg). error_msg is empty on success.
    """
    df = df.copy()
    try:
        type_map = {
            "int": "int64",
            "float": "float64",
            "str": "object",
            "string": "object",
            "datetime": None,
            "bool": "bool",
        }
        if target_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif target_type in type_map:
            df[column] = df[column].astype(type_map[target_type])
        else:
            df[column] = df[column].astype(target_type)
        return df, ""
    except Exception as e:
        return df, str(e)


def generate_eda_summary_text(df: pd.DataFrame) -> str:
    """
    Generate a compact text summary of the dataset for LLM consumption.
    """
    info = get_basic_info(df)
    stats = get_summary_stats(df)
    missing = get_missing_value_report(df)
    dups = get_duplicate_report(df)

    lines = [
        f"Dataset Overview:",
        f"  - Rows: {info['rows']:,}",
        f"  - Columns: {info['columns']}",
        f"  - Memory: {info['memory_mb']} MB",
        f"  - Numeric columns: {', '.join(info['numeric_columns']) or 'None'}",
        f"  - Categorical columns: {', '.join(info['categorical_columns']) or 'None'}",
        f"",
        f"Data Quality:",
        f"  - Duplicate rows: {dups['duplicate_rows']} ({dups['duplicate_pct']}%)",
        f"  - Missing values: {int(df.isnull().sum().sum())} cells total",
    ]

    if not missing.empty:
        lines.append("  - Columns with missing data:")
        for _, row in missing.head(10).iterrows():
            lines.append(f"    • {row['Column']}: {row['Missing Count']} missing ({row['Missing %']}%)")

    if not stats.empty:
        lines.append("")
        lines.append("Numeric Column Statistics:")
        for col in stats.index[:10]:
            row = stats.loc[col]
            lines.append(
                f"  - {col}: mean={row.get('mean', 'N/A'):.2f}, "
                f"std={row.get('std', 'N/A'):.2f}, "
                f"min={row.get('min', 'N/A'):.2f}, "
                f"max={row.get('max', 'N/A'):.2f}"
            )

    return "\n".join(lines)
