"""
config.py — Global Configuration & Constants
AI Data Analyst Assistant
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# App Identity
# ──────────────────────────────────────────
APP_NAME = "AI Data Analyst Assistant"
APP_VERSION = "1.0.0"
APP_ICON = "🤖"
APP_DESCRIPTION = "Enterprise-grade AI-powered data analysis platform"

# ──────────────────────────────────────────
# Directory Paths (Relative — works on Streamlit Cloud)
# ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"

# Auto-create all required directories
for _dir in [ASSETS_DIR, UPLOADS_DIR, DATA_DIR, REPORTS_DIR, DATABASE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────
# Database
# ──────────────────────────────────────────
DB_PATH = DATABASE_DIR / "app.db"

# ──────────────────────────────────────────
# API Keys — Streamlit Cloud secrets first, then env vars
# ──────────────────────────────────────────
def get_groq_api_key() -> str:
    """Retrieve Groq API key from Streamlit secrets or environment."""
    try:
        return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

# ──────────────────────────────────────────
# LLM Defaults
# ──────────────────────────────────────────
DEFAULT_MODEL = "llama-3.3-70b-versatile"
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096

# ──────────────────────────────────────────
# RAG / Embedding Defaults
# ──────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AVAILABLE_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
]
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5

# ──────────────────────────────────────────
# File Upload Limits
# ──────────────────────────────────────────
ALLOWED_EXTENSIONS = ["csv", "xlsx", "xls", "pdf"]
MAX_FILE_SIZE_MB = 50

# ──────────────────────────────────────────
# UI / Theme
# ──────────────────────────────────────────
THEME_DARK = "dark"
THEME_LIGHT = "light"
DEFAULT_THEME = THEME_DARK

# ──────────────────────────────────────────
# Navigation Pages
# ──────────────────────────────────────────
PAGES = {
    "home": "🏠 Home",
    "upload": "📂 Upload Data",
    "chat": "💬 Chat with AI",
    "analytics": "📊 Analytics Dashboard",
    "insights": "💡 Business Insights",
    "reports": "📋 Reports",
    "history": "🕐 Chat History",
    "settings": "⚙️ Settings",
    "developer": "👨‍💻 Developer",
}

# ──────────────────────────────────────────
# SQL Mode
# ──────────────────────────────────────────
SQLITE_DATASETS_DIR = DATA_DIR / "sqlite"
SQLITE_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────
# Chart Types
# ──────────────────────────────────────────
CHART_TYPES = [
    "Bar Chart",
    "Line Chart",
    "Pie Chart",
    "Histogram",
    "Scatter Plot",
    "Area Chart",
    "Box Plot",
    "Heatmap",
    "Treemap",
    "Sunburst",
]
