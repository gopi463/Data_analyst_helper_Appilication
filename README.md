# 🤖 AI Data Analyst Assistant v2.0

> Enterprise-grade AI-powered data analytics platform with Hybrid RAG, Streaming, Conversation Memory, and Web Search Fallback

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange)](https://groq.com)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Search-BM25%20%2B%20FAISS-purple)](https://github.com/dorianbrown/rank_bm25)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔐 Authentication | Secure login/register with bcrypt password hashing |
| 📂 File Upload | CSV, Excel (.xlsx/.xls), PDF — multi-file support |
| 💬 AI Chat | Hybrid RAG Q&A, SQL Mode, Pandas Mode with **streaming responses** |
| 🧠 Conversation Memory | Follow-up questions work naturally (last 4 exchanges remembered) |
| 🔍 Hybrid Search | BM25 keyword search + FAISS semantic search fused with RRF |
| 🎯 Reranking | Score-fusion reranker for improved answer quality |
| 🌐 Web Search Fallback | Auto-fallback to DuckDuckGo when documents don't contain the answer |
| 📊 Analytics | Full EDA, missing values, outlier detection, correlation |
| 🧹 Data Cleaning | Remove duplicates, fill missing, change types, rename columns |
| 📈 Charts | 10+ Plotly chart types with AI auto-suggestion |
| 💡 Insights | Auto KPIs, top performers, trend analysis, AI recommendations |
| 📋 Reports | Professional PDF reports with executive summaries |
| 👍 Feedback | Thumbs-up/thumbs-down buttons on every AI response |
| 🕐 Chat History | Per-user conversation storage with search and delete |
| ⚙️ Settings | Model, temperature, top-k, chunk size, embedding model |
| 🧪 Tests | Unit tests for core modules (pytest) |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌───────────────────┐  │
│  │  Upload  │  │   Chat   │  │ Analytics  │  │ Insights/Reports  │  │
│  └────┬─────┘  └────┬─────┘  └────────────┘  └───────────────────┘  │
└───────│─────────────│────────────────────────────────────────────────┘
        │             │
        ▼             ▼
┌───────────────┐   ┌─────────────────────────────────────────────────┐
│  Document     │   │              Hybrid RAG Pipeline                 │
│  Ingestion    │   │                                                   │
│               │   │  ┌─────────────┐    ┌──────────────────────┐    │
│  loader.py    │   │  │  BM25 Index  │    │   FAISS Vector Store  │   │
│  chunker.py   │───┼─▶│ (rank-bm25) │    │ (SentenceTransformer) │   │
│  embedder.py  │   │  └──────┬──────┘    └──────────┬───────────┘    │
└───────────────┘   │         │                       │                │
                    │         └──────────┬────────────┘                │
                    │                    ▼                              │
                    │         ┌──────────────────┐                     │
                    │         │  RRF Fusion +    │                     │
                    │         │  Score Reranker  │                     │
                    │         └────────┬─────────┘                     │
                    │                  │  low confidence?              │
                    │                  ├──────────────▶ Web Search     │
                    │                  │               (DuckDuckGo)    │
                    │                  ▼                               │
                    │    ┌─────────────────────────────┐              │
                    │    │  Groq LLM (Llama-3.3-70B)   │              │
                    │    │  + Conversation Memory       │ ◀─ Streaming │
                    │    └─────────────────────────────┘              │
                    └─────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                     Persistence Layer                              │
│   SQLite (users, chat_history, feedback, files, settings)          │
│   FAISS Index + BM25 metadata (per-user, on disk)                 │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd RAG_SYSTEM
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 5. Run the application
```bash
streamlit run app.py
```

### 6. Run Tests
```bash
python -m pytest tests/ -v --tb=short
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push this project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app pointing to `app.py`
4. In **App Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy!

---

## 🔧 Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `llama-3.3-70b-versatile` | Groq LLM model |
| `temperature` | `0.1` | LLM creativity (0=precise, 1=creative) |
| `top_k` | `5` | Number of chunks to retrieve per query |
| `chunk_size` | `600` | Text chunk size (characters) |
| `embedding_model` | `all-MiniLM-L6-v2` | SentenceTransformer model name |

---

## 📁 Project Structure

```
RAG_SYSTEM/
├── app.py                  # Main Streamlit entry point (routing + 8 pages)
├── config.py               # Global configuration & constants
├── database.py             # SQLite ORM (users, history, feedback, files, reports)
├── auth.py                 # Authentication (login, register, bcrypt)
├── loader.py               # File loader (CSV, Excel, PDF)
├── chunker.py              # Smart text chunker with sentence-boundary awareness
├── embedder.py             # SentenceTransformer embedding engine (cached)
├── vector_store.py         # FAISS vector database (per-user isolation)
├── retriever.py            # ★ Hybrid BM25+FAISS retriever + RRF + reranking
├── web_search.py           # ★ DuckDuckGo web search fallback
├── llm.py                  # ★ Groq LLM client (streaming + conversation memory)
├── analytics.py            # EDA, cleaning, outlier detection
├── charts.py               # Plotly chart builder (10+ types)
├── insights.py             # Business KPIs and insight cards
├── report_generator.py     # PDF report generation with ReportLab
├── history.py              # Chat history manager
├── utils.py                # Helpers, SQL/Pandas execution, UI components
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── tests/
│   ├── test_chunker.py     # ★ Unit tests for chunker
│   ├── test_retriever.py   # ★ Unit tests for hybrid retriever
│   ├── test_llm.py         # ★ Unit tests for LLM helpers
│   ├── test_analytics.py   # ★ Unit tests for analytics
│   └── test_database.py    # ★ Unit tests for database
├── assets/
│   └── style.css           # Dark glassmorphism CSS theme
├── uploads/                # Uploaded files storage
├── data/                   # FAISS indexes (per user)
├── reports/                # Generated PDF reports (per user)
└── database/               # SQLite database files
```
> ★ = New or significantly upgraded in v2.0

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| AI / LLM | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Semantic Search | FAISS (Facebook AI Similarity Search) |
| Keyword Search | BM25 (`rank-bm25`) |
| Search Fusion | Reciprocal Rank Fusion (RRF) |
| Web Fallback | DuckDuckGo Search (`duckduckgo-search`) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Database | SQLite |
| Authentication | bcrypt |
| PDF Generation | ReportLab |
| File Parsing | pypdf, openpyxl |
| Testing | pytest |

---

## 📸 Pages Overview

- **🏠 Home** — Dashboard with KPI cards, file history, quick actions
- **📂 Upload Data** — Multi-file drag & drop with 5-step progress bar and BM25+FAISS indexing
- **💬 Chat with AI** — Streaming RAG Q&A with conversation memory, web fallback, and 👍/👎 feedback
- **📊 Analytics Dashboard** — Full EDA + data cleaning + custom chart builder
- **💡 Business Insights** — KPIs, top performers, trend charts, AI recommendations
- **📋 Reports** — PDF generation with executive summary, export CSV/Excel
- **🕐 Chat History** — Search, browse, continue, delete conversation history
- **⚙️ Settings** — Model, temperature, chunk size, embedding model, API key

---

## 🔑 Getting a Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys**
4. Create a new key
5. Copy and paste it into your `.env` file or Streamlit Cloud secrets

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_chunker.py -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests: `python -m pytest tests/`
4. Commit: `git commit -m "Add my feature"`
5. Push: `git push origin feature/my-feature`
6. Open a Pull Request

---

## 📄 License

MIT License — feel free to use this in your own projects!
