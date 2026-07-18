# 🤖 AI Data Analyst Assistant

> Enterprise-grade AI-powered data analytics platform built with Python + Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔐 Authentication | Secure login/register with bcrypt password hashing |
| 📂 File Upload | CSV, Excel (.xlsx/.xls), PDF — multi-file support |
| 💬 AI Chat | RAG Q&A, SQL Mode, Pandas Mode |
| 📊 Analytics | Full EDA, missing values, outlier detection, correlation |
| 🧹 Data Cleaning | Remove duplicates, fill missing, change types, rename columns |
| 📈 Charts | 10+ Plotly chart types with auto-suggestion |
| 💡 Insights | Auto KPIs, top performers, trend analysis, AI recommendations |
| 📋 Reports | Professional PDF reports with executive summaries |
| 🕐 Chat History | Per-user conversation storage with search and delete |
| ⚙️ Settings | Model, temperature, top-k, chunk size, embedding model |

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

## 📁 Project Structure

```
RAG_SYSTEM/
├── app.py                  # Main Streamlit entry point (routing + 8 pages)
├── config.py               # Global configuration & constants
├── database.py             # SQLite ORM (users, history, files, reports)
├── auth.py                 # Authentication (login, register, bcrypt)
├── loader.py               # File loader (CSV, Excel, PDF)
├── chunker.py              # Text chunker for RAG
├── embedder.py             # SentenceTransformer embedding engine
├── vector_store.py         # FAISS vector database
├── retriever.py            # RAG retriever with citations
├── llm.py                  # Groq LLM client (RAG, SQL, Pandas, Insights)
├── analytics.py            # EDA, cleaning, outlier detection
├── charts.py               # Plotly chart builder (10+ types)
├── insights.py             # Business KPIs and insight cards
├── report_generator.py     # PDF report generation with ReportLab
├── history.py              # Chat history manager
├── utils.py                # Helpers, SQL/Pandas execution, UI components
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── assets/
│   └── style.css           # Dark glassmorphism CSS theme
├── uploads/                # Uploaded files storage
├── data/                   # FAISS indexes (per user)
├── reports/                # Generated PDF reports (per user)
└── database/               # SQLite database files
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| AI / LLM | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Database | SQLite |
| Authentication | bcrypt |
| PDF Generation | ReportLab |
| File Parsing | pypdf, openpyxl |

---

## 🔑 Getting a Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys**
4. Create a new key
5. Copy and paste it into your `.env` file or Streamlit Cloud secrets

---

## 📸 Pages Overview

- **🏠 Home** — Dashboard with KPI cards, file history, quick actions
- **📂 Upload Data** — Multi-file drag & drop with EDA preview and RAG indexing
- **💬 Chat with AI** — RAG Q&A, SQL mode, Pandas code execution with citations
- **📊 Analytics Dashboard** — Full EDA + data cleaning + custom chart builder
- **💡 Business Insights** — KPIs, top performers, trend charts, AI recommendations
- **📋 Reports** — PDF generation with executive summary, export CSV/Excel
- **🕐 Chat History** — Search, browse, continue, delete conversation history
- **⚙️ Settings** — Model, temperature, chunk size, embedding model, API key

---

## 📄 License

MIT License — feel free to use this in your own projects!
