# 📊 Data Analyst Helper Application

> **An enterprise-grade, AI-powered analytics platform featuring Hybrid RAG (BM25 + FAISS), Real-Time Streaming, Multi-Turn Conversation Memory, Automated Exploratory Data Analysis, Dynamic Visualizations, and Intelligent Web Fallback.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Lexical-BM25Okapi-8A2BE2?style=for-the-badge)](https://github.com/dorianbrown/rank_bm25)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 🌟 Overview

The **Data Analyst Helper Application** transforms raw spreadsheets, documents, and business data into actionable business intelligence. Built on a state-of-the-art **Hybrid Retrieval-Augmented Generation (RAG)** pipeline, it pairs dense semantic vector search with lexical keyword retrieval to achieve high answer precision across structured and unstructured files (CSV, Excel, PDF).

Whether you need automated exploratory data analysis, interactive data cleaning, real-time streaming Q&A, SQL/Pandas code generation, 10+ interactive Plotly charts, or publication-grade executive PDF reports, this application provides an end-to-end data analytics workflow within a sleek, glassmorphic dark-themed interface.

---

## 🚀 Key Features

| Category | Highlights |
|---|---|
| 🔐 **Authentication & Security** | Per-user isolation, secure registration/login using `bcrypt` password hashing, private session state, and user-specific document storage. |
| 📂 **Multi-Format Ingestion** | Drag-and-drop support for **CSV**, **Excel (.xlsx, .xls)**, and **PDF** documents with multi-file merging and sentence-boundary text chunking. |
| 🧠 **Hybrid RAG Pipeline** | Dense semantic vectors (**FAISS** with `sentence-transformers`) + Sparse keyword matching (**BM25Okapi**), fused via **Reciprocal Rank Fusion (RRF)** and confidence score reranking. |
| 💬 **Streaming Conversational AI** | Ultra-low latency streaming responses powered by **Groq** (`llama-3.3-70b-versatile`), equipped with 4-exchange conversational memory and query rewriting. |
| 🌐 **Intelligent Web Fallback** | Automatic fallback to **DuckDuckGo Web Search** when uploaded documents do not contain the answer or confidence falls below threshold. |
| 💻 **SQL & Pandas Execution** | Ask natural language questions that automatically compile into executable SQL queries (SQLite in-memory) or Pandas operations with live execution and verification. |
| 📊 **Deep Exploratory Data Analysis** | Instant summary statistics, null value matrices, skewness/kurtosis assessment, IQR outlier detection, and correlation heatmaps. |
| 🧹 **Interactive Data Cleaning** | Deduplicate rows, impute missing values (mean, median, mode, constant), cast data types, drop sparse columns, and rename features. |
| 📈 **Dynamic Visualizations** | 10+ Plotly chart types (Bar, Line, Scatter, Histogram, Box, Heatmap, Pie, Treemap, Violin, Area) with an automated AI chart recommender. |
| 💡 **Automated Business Insights** | Instant KPI metric extraction, top-performer rankings, trend discovery, anomalies, and AI-generated strategic recommendations. |
| 📋 **Executive PDF Reports** | One-click publication-grade PDF report generator powered by ReportLab, complete with executive summaries, KPI tables, and data exports to CSV/Excel. |
| 👨‍💻 **Developer & Recruiter Hub** | Integrated developer profile highlighting technical specializations and a direct recruiter contact inquiry form. |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       Streamlit Glassmorphic UI (app.py)                   │
│   ┌──────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐  │
│   │ 🏠 Dashboard │   │  📂 Upload   │   │  💬 Chat   │   │  📊 Analytics│  │
│   ├──────────────┤   ├──────────────┤   ├────────────┤   ├──────────────┤  │
│   │  💡 Insights │   │  📋 Reports  │   │  ⚙️ Settings│  │  👨‍💻 Developer│  │
│   └──────────────┘   └──────────────┘   └────────────┘   └──────────────┘  │
└───────────────────────┬──────────────────────┬─────────────────────────────┘
                        │                      │
                        ▼                      ▼
┌────────────────────────────────┐   ┌───────────────────────────────────────┐
│     Document Ingestion         │   │         Hybrid RAG Pipeline           │
│                                │   │                                       │
│  • loader.py (CSV, XLSX, PDF)  │   │  ┌──────────────┐   ┌──────────────┐  │
│  • chunker.py (Sentence Split) │   │  │  FAISS Index │   │  BM25 Index  │  │
│  • embedder.py (all-MiniLM-L6) │───┼─▶│ (Dense/Sem.) │   │(Lexical/Keyw)│  │
└────────────────────────────────┘   │  └──────┬───────┘   └──────┬───────┘  │
                                     │         │                  │          │
                                     │         └────────┬─────────┘          │
                                     │                  ▼                    │
                                     │      ┌────────────────────────┐       │
                                     │      │ Reciprocal Rank Fusion │       │
                                     │      │   + Score Reranker     │       │
                                     │      └───────────┬────────────┘       │
                                     │                  │                    │
                                     │        Low Score?│ Yes                │
                                     │                  ├──────▶ Web Search  │
                                     │                  │        (DuckDuckGo)│
                                     │                  ▼                    │
                                     │      ┌────────────────────────┐       │
                                     │      │ Groq Cloud Engine      │       │
                                     │      │ llama-3.3-70b-versatile│       │
                                     │      │ + Multi-Turn Memory    │       │
                                     │      └───────────┬────────────┘       │
                                     │                  │ Streaming          │
                                     │                  ▼                    │
                                     │        User Interface Display         │
                                     └───────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                           Persistence Layer                                │
│   • SQLite Database (database.py) — Users, Sessions, History, Feedback     │
│   • Vector Storage (vector_store.py) — Per-user FAISS index & BM25 corpus  │
│   • Generated Artifacts — Reports (.pdf), Exported Data (.csv, .xlsx)      │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack & Dependencies

- **Frontend & App Framework:** [Streamlit](https://streamlit.io)
- **Large Language Models:** [Groq Cloud API](https://groq.com)
  - `llama-3.3-70b-versatile` (Default High-Reasoning Model)
  - `llama-3.1-8b-instant` (Fast Sub-Second Fallback)
  - `openai/gpt-oss-120b` / `openai/gpt-oss-20b`
- **Embedding Model:** [SentenceTransformers](https://sbert.net) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Search Engine:** [FAISS CPU](https://github.com/facebookresearch/faiss)
- **Keyword Search Engine:** [Rank-BM25](https://github.com/dorianbrown/rank_bm25)
- **Web Search Integration:** [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)
- **Data Analysis & Modeling:** Pandas, NumPy, SciPy, Scikit-learn
- **Data Visualization:** Plotly Graph Objects & Plotly Express
- **PDF Generation Engine:** ReportLab
- **Authentication & Hashing:** bcrypt
- **Database:** SQLite3

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/gopi463/Data_analyst_helper_Appilication.git
cd Data_analyst_helper_Appilication
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory (or duplicate `.env.example`):
```bash
cp .env.example .env
```
Add your **Groq API Key**:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
```
> *(You can also leave `.env` empty and provide your Groq API Key directly in the app's Settings page or sidebar at runtime!)*

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 Running the Test Suite

The project includes unit tests covering the chunker, hybrid retriever, Groq LLM integration, data analytics, and database layers:

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run with summary of test outputs
python -m pytest tests/ -v --tb=short

# Run specific module tests
python -m pytest tests/test_retriever.py -v
python -m pytest tests/test_chunker.py -v
```

---

## 📁 Repository Structure

```
Data_analyst_helper_Appilication/
├── app.py                  # Main application router and page renderer
├── config.py               # Application settings, live Groq models & constants
├── database.py             # SQLite persistence, schema migrations, and user data
├── auth.py                 # User authentication, registration, and bcrypt hashing
├── loader.py               # Ingestion parser for CSV, Excel (.xlsx/.xls), and PDF
├── chunker.py              # Sentence-aware text chunking with overlap
├── embedder.py             # SentenceTransformer embeddings engine with caching
├── vector_store.py         # FAISS vector store creation, index disk persistence
├── retriever.py            # Hybrid BM25 + FAISS search, RRF fusion & reranker
├── web_search.py           # DuckDuckGo web search fallback provider
├── llm.py                  # Groq API streaming client with conversation memory
├── analytics.py            # Automated EDA, summary statistics, outlier detection
├── charts.py               # 10+ interactive Plotly chart templates & AI recommender
├── insights.py             # KPI calculation, trend detection, AI recommendations
├── report_generator.py     # Publication-grade PDF report compiler (ReportLab)
├── history.py              # Chat history search, conversation management
├── utils.py                # SQL sandbox, Pandas runner, UI helpers & formatters
├── requirements.txt        # Pinned production dependencies
├── .env.example            # Environment configuration template
├── .gitignore              # Git exclusions for environments, databases, caches
├── assets/
│   └── style.css           # Glassmorphism dark mode stylesheet
├── tests/
│   ├── test_chunker.py     # Chunker unit tests
│   ├── test_retriever.py   # Hybrid search & RRF unit tests
│   ├── test_llm.py         # LLM prompt builder and client tests
│   ├── test_analytics.py   # EDA and data cleaning tests
│   └── test_database.py    # Database schema & migrations tests
├── data/                   # User-isolated FAISS vector index files (.gitignore)
├── database/               # Local SQLite database instances (.gitignore)
├── reports/                # Generated user PDF reports (.gitignore)
└── uploads/                # User uploaded datasets (.gitignore)
```

---

## 🌐 Deploying to Streamlit Community Cloud

1. Fork or push this repository to your GitHub account (`gopi463/Data_analyst_helper_Appilication`).
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select your repository, branch (`main`), and main file path (`app.py`).
4. Click **Advanced settings...** and add your secrets:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_api_key"
   ```
5. Click **Deploy!**

---

## 👨‍💻 Author & Developer

**Gopi Chand Pasam**  
*AI & RAG Engineer and Data Analyst*

- 🐙 **GitHub:** [@gopi463](https://github.com/gopi463)
- 💼 **LinkedIn:** [Gopi Chand Pasam](https://linkedin.com)
- 📬 **Email:** [gopipasam93@gmail.com](mailto:gopipasam93@gmail.com)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
