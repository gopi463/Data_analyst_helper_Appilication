"""
utils.py — Utility Functions & Helpers
AI Data Analyst Assistant
"""

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import re
import io
from datetime import datetime
from typing import Any, Optional
from contextlib import contextmanager
from config import SQLITE_DATASETS_DIR


# ──────────────────────────────────────────
# Session State Initializer
# ──────────────────────────────────────────
def init_session_state() -> None:
    """Initialize all required session state keys with default values."""
    defaults = {
        "authenticated": False,
        "user": None,
        "user_id": None,
        "current_page": "home",
        "chat_messages": [],
        "session_id": None,
        "uploaded_data": None,      # List of load results
        "current_df": None,         # Active DataFrame
        "vector_store": None,       # FAISS VectorStore
        "chunks": None,             # RAG chunks
        "file_names": [],           # Uploaded file names
        "current_sql_db": None,     # SQLite path for SQL mode
        "sql_table_name": "data_table",
        "groq_api_key": "",
        "settings": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ──────────────────────────────────────────
# DataFrame → SQLite (SQL Mode)
# ──────────────────────────────────────────
def load_df_to_sqlite(df: pd.DataFrame, user_id: int, table_name: str = "data_table") -> str:
    """
    Load a DataFrame into a per-user SQLite database for SQL queries.
    Returns the SQLite file path.
    """
    db_path = SQLITE_DATASETS_DIR / f"user_{user_id}_dataset.db"
    conn = sqlite3.connect(str(db_path))
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return str(db_path)


def execute_sql_query(db_path: str, sql: str) -> tuple[Optional[pd.DataFrame], str]:
    """
    Execute a SQL query against a SQLite database.
    Returns (result_df, error_message).
    """
    # Safety: block destructive operations
    forbidden = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\b", re.IGNORECASE)
    if forbidden.search(sql):
        return None, "⚠️ Destructive SQL operations are not permitted."
    try:
        conn = sqlite3.connect(db_path)
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return result, ""
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────
# Pandas Code Execution
# ──────────────────────────────────────────
def execute_pandas_code(code: str, df: pd.DataFrame) -> tuple[Any, str]:
    """
    Safely execute LLM-generated pandas code.
    Returns (result, error_message).
    """
    # Safety: block dangerous operations
    forbidden_patterns = [
        r"\bos\b", r"\bsubprocess\b", r"\bopen\b", r"\bexec\b",
        r"\beval\b", r"__import__", r"\bshutil\b", r"\bsocket\b",
    ]
    for pat in forbidden_patterns:
        if re.search(pat, code):
            return None, f"⚠️ Unsafe operation detected in generated code: '{pat}'"

    try:
        local_vars = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, {"__builtins__": {}}, local_vars)
        result = local_vars.get("result", None)
        return result, ""
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────
# Number / Value Formatters
# ──────────────────────────────────────────
def format_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}"


def format_currency(value: float) -> str:
    """Format value as currency string."""
    return f"${format_number(value)}"


def format_percent(value: float) -> str:
    """Format value as percentage string."""
    return f"{value:+.1f}%" if value != 0 else "0.0%"


def truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate long text with ellipsis."""
    return text if len(text) <= max_len else text[:max_len] + "…"


# ──────────────────────────────────────────
# Time Helpers
# ──────────────────────────────────────────
def time_ago(dt_str: str) -> str:
    """Convert a datetime string to a human-readable relative time."""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt_str
    delta = datetime.utcnow() - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        return f"{seconds // 60}m ago"
    elif seconds < 86400:
        return f"{seconds // 3600}h ago"
    else:
        return f"{seconds // 86400}d ago"


# ──────────────────────────────────────────
# Streamlit UI Helpers
# ──────────────────────────────────────────
def success_box(message: str) -> None:
    st.markdown(
        f'<div style="background:#0d2b1f;border:1px solid #2ea043;border-radius:8px;padding:12px 16px;'
        f'color:#2ea043;font-weight:600;">✅ {message}</div>',
        unsafe_allow_html=True,
    )


def error_box(message: str) -> None:
    st.markdown(
        f'<div style="background:#2d1117;border:1px solid #f85149;border-radius:8px;padding:12px 16px;'
        f'color:#f85149;font-weight:600;">❌ {message}</div>',
        unsafe_allow_html=True,
    )


def info_box(message: str) -> None:
    st.markdown(
        f'<div style="background:#0d1f2d;border:1px solid #388bfd;border-radius:8px;padding:12px 16px;'
        f'color:#79c0ff;font-weight:500;">ℹ️ {message}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str = "📊", color: str = "#667eea", delta: str = None) -> None:
    """Render a styled KPI metric card."""
    delta_html = f'<div style="color:#3fb950;font-size:0.8rem;margin-top:4px;">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(102,126,234,0.15),rgba(118,75,162,0.1));
        border:1px solid rgba(102,126,234,0.3);border-radius:12px;padding:16px 20px;
        text-align:center;height:100%;">
        <div style="font-size:1.8rem;margin-bottom:4px;">{icon}</div>
        <div style="color:#8892a4;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">{label}</div>
        <div style="color:{color};font-size:1.5rem;font-weight:800;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def insight_card(title: str, value: str, subtitle: str, icon: str, color: str) -> None:
    """Render a styled business insight card."""
    st.markdown(f"""
    <div style="background:rgba(22,27,34,0.8);border:1px solid rgba(255,255,255,0.08);
        border-left:4px solid {color};border-radius:10px;padding:14px 18px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.4rem;">{icon}</span>
            <div>
                <div style="color:#8892a4;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;">{title}</div>
                <div style="color:#e6edf3;font-size:1.1rem;font-weight:700;">{value}</div>
                <div style="color:#6e7681;font-size:0.8rem;">{subtitle}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    """Render a section header with optional subtitle."""
    sub_html = f'<p style="color:#8892a4;font-size:0.95rem;margin-top:4px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <h2 style="color:#e6edf3;font-size:1.6rem;font-weight:700;margin:0;">{title}</h2>
        {sub_html}
        <div style="height:3px;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:3px;margin-top:8px;width:80px;"></div>
    </div>
    """, unsafe_allow_html=True)


def no_data_placeholder(message: str = "No data uploaded yet.") -> None:
    """Render a placeholder when no data is available."""
    st.markdown(f"""
    <div style="text-align:center;padding:4rem 2rem;background:rgba(22,27,34,0.5);
        border:2px dashed rgba(102,126,234,0.3);border-radius:16px;margin:2rem 0;">
        <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
        <h3 style="color:#8892a4;">{message}</h3>
        <p style="color:#6e7681;font-size:0.9rem;">
            Go to <strong>Upload Data</strong> to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
