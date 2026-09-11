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
# Live model list — updated Sept 2026.
# llama3-70b-8192, llama-3.1-70b-versatile, gemma2-9b-it, mixtral-8x7b-32768
# are ALL decommissioned. Use get_available_groq_models() at runtime for latest.
DEFAULT_MODEL = "llama-3.3-70b-versatile"   # current primary production model
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",   # primary — 70B, best quality
    "llama-3.1-8b-instant",      # fast & cheap fallback
    "openai/gpt-oss-20b",        # OpenAI-compatible fallback
    "openai/gpt-oss-120b",       # large OpenAI-compatible fallback
]
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096


def get_available_groq_models(api_key: str = "") -> list:
    """
    Fetch the live list of text-generation models from Groq API.
    Falls back to AVAILABLE_MODELS if the API call fails (network issue,
    invalid key, etc.).
    Only returns chat/text models — filters out audio/whisper/speech models.
    """
    key = api_key or get_groq_api_key()
    if not key:
        return AVAILABLE_MODELS
    try:
        from groq import Groq
        client = Groq(api_key=key)
        response = client.models.list()
        # Filter to text-generation chat models only
        skip_keywords = ["whisper", "tts", "distil", "guard", "vision", "playai", "speech"]
        models = [
            m.id for m in response.data
            if not any(kw in m.id.lower() for kw in skip_keywords)
        ]
        if models:
            return sorted(models)
    except Exception:
        pass
    return AVAILABLE_MODELS

# ──────────────────────────────────────────
# RAG / Embedding Defaults
# ──────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AVAILABLE_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
]
DEFAULT_CHUNK_SIZE = 1000      # Larger chunks = fewer total chunks = faster embedding
DEFAULT_CHUNK_OVERLAP = 150   # Slightly bigger overlap keeps context quality
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
