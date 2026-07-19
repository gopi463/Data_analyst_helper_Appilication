"""
app.py — Main Streamlit Application
AI Data Analyst Assistant

Entry point routing all 8 pages with full authentication gate.
"""

import streamlit as st


import pandas as pd
import numpy as np
import os
import uuid
from pathlib import Path
from datetime import datetime

# ── Local Modules ─────────────────────────────────────────────────────────────
import database as db
import auth
import history as hist
import analytics
import charts as ch
import insights as ins
import utils
import llm as llm_module
from loader import load_multiple_files, get_file_size_str
from chunker import chunk_documents
from embedder import embed_chunks
from vector_store import VectorStore, build_vector_store
from retriever import retrieve_with_context, format_sources, BM25Retriever
from web_search import web_search, format_web_results_as_context, format_web_sources
from report_generator import generate_pdf_report, export_csv, export_excel
from config import (
    APP_NAME, APP_ICON, PAGES, DEFAULT_TOP_K, DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP, DEFAULT_MODEL, DEFAULT_TEMPERATURE,
    AVAILABLE_MODELS, AVAILABLE_EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL,
    ASSETS_DIR
)

# ──────────────────────────────────────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize DB and session
db.init_db()
utils.init_session_state()

# Load CSS
def _load_css():
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()


# ──────────────────────────────────────────────────────────────────────────────
# Auth Gate
# ──────────────────────────────────────────────────────────────────────────────
if not auth.is_logged_in():
    auth.render_auth_page()
    st.stop()

# Helpers
user = st.session_state["user"]
user_id = st.session_state["user_id"]
settings = db.get_user_settings(user_id)
st.session_state["settings"] = settings
api_key = st.session_state.get("groq_api_key") or utils.init_session_state() or os.getenv("GROQ_API_KEY", "")

try:
    import streamlit as _st
    _api = _st.secrets.get("GROQ_API_KEY", "")
    if _api:
        api_key = _api
except Exception:
    pass

if not api_key:
    api_key = st.session_state.get("groq_api_key", "")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:2.5rem;">🤖</div>
            <div style="font-size:1rem;font-weight:700;
                background:linear-gradient(135deg,#667eea,#764ba2);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                AI Data Analyst
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # User info
        display_name = user.get("full_name") or user.get("username", "User")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
            background:rgba(102,126,234,0.1);border-radius:10px;margin-bottom:12px;">
            <div style="width:36px;height:36px;border-radius:50%;
                background:linear-gradient(135deg,#667eea,#764ba2);
                display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:1rem;color:white;">
                {display_name[0].upper()}
            </div>
            <div>
                <div style="color:#e6edf3;font-weight:600;font-size:0.9rem;">{display_name}</div>
                <div style="color:#8892a4;font-size:0.75rem;">@{user.get("username","")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        nav_items = [
            ("home",     "🏠", "Home"),
            ("upload",   "📂", "Upload Data"),
            ("chat",     "💬", "Chat with AI"),
            ("analytics","📊", "Analytics Dashboard"),
            ("insights", "💡", "Business Insights"),
            ("reports",  "📋", "Reports"),
            ("history",  "🕐", "Chat History"),
            ("settings", "⚙️", "Settings"),
            ("developer", "👨‍💻", "Developer"),
        ]

        for page_id, icon, label in nav_items:
            is_active = st.session_state.get("current_page") == page_id
            bg = "rgba(102,126,234,0.2)" if is_active else "transparent"
            color = "#e6edf3" if is_active else "#8892a4"
            border = "border-left:3px solid #667eea;" if is_active else "border-left:3px solid transparent;"
            if st.sidebar.button(
                f"{icon} {label}",
                key=f"nav_{page_id}",
                use_container_width=True,
            ):
                st.session_state["current_page"] = page_id
                st.rerun()

        st.markdown("---")

        # API Key input if not set
        if not api_key:
            new_key = st.sidebar.text_input("🔑 Groq API Key", type="password", key="sidebar_api_key_input",
                                             placeholder="gsk_...")
            if new_key:
                st.session_state["groq_api_key"] = new_key
                st.rerun()
        else:
            st.sidebar.markdown(
                '<div style="color:#3fb950;font-size:0.8rem;padding:4px 8px;">✅ API Key configured</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Data status
        df = st.session_state.get("current_df")
        if df is not None:
            st.sidebar.markdown(f"""
            <div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);
                border-radius:8px;padding:10px 12px;font-size:0.8rem;color:#3fb950;">
                📊 <strong>Data Loaded</strong><br>
                {len(df):,} rows × {len(df.columns)} cols
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div style="background:rgba(240,147,251,0.08);border:1px solid rgba(240,147,251,0.2);
                border-radius:8px;padding:10px 12px;font-size:0.8rem;color:#f093fb;">
                📭 No data loaded
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.sidebar.button("🚪 Logout", use_container_width=True):
            auth.logout()
            st.rerun()


render_sidebar()
current_page = st.session_state.get("current_page", "home")


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: HOME ████████████████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_home():
    utils.section_header(
        f"Welcome back, {user.get('full_name') or user.get('username')}! 👋",
        "Your AI-powered data analytics command center"
    )

    stats = db.get_user_stats(user_id)
    df = st.session_state.get("current_df")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        utils.metric_card("Total Sessions", str(stats["total_sessions"]), "🗣️", "#667eea")
    with c2:
        utils.metric_card("Messages Sent", str(stats["total_messages"]), "💬", "#764ba2")
    with c3:
        utils.metric_card("Files Uploaded", str(stats["total_files"]), "📂", "#f093fb")
    with c4:
        utils.metric_card("Reports Generated", str(stats["total_reports"]), "📋", "#4facfe")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 📊 Data Overview")
        if df is not None:
            kpis = ins.extract_kpis(df)
            kpi_cols = st.columns(min(4, len(kpis)))
            for i, kpi in enumerate(kpis[:4]):
                with kpi_cols[i]:
                    formatted = ins.format_kpi_value(kpi["value"], kpi["format"])
                    utils.metric_card(kpi["label"], formatted, kpi["icon"], "#667eea")

            st.markdown("")
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        else:
            utils.no_data_placeholder("Upload your data to see analytics here!")

    with col_right:
        st.markdown("### 🚀 Quick Actions")
        if st.button("📂 Upload New Data", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "upload"
            st.rerun()
        if st.button("💬 Start AI Chat", use_container_width=True):
            st.session_state["current_page"] = "chat"
            st.rerun()
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state["current_page"] = "analytics"
            st.rerun()
        if st.button("💡 Business Insights", use_container_width=True):
            st.session_state["current_page"] = "insights"
            st.rerun()
        if st.button("📋 Generate Report", use_container_width=True):
            st.session_state["current_page"] = "reports"
            st.rerun()

        st.markdown("### 📁 Recent Files")
        files = db.get_user_files(user_id)
        if files:
            for f in files[:5]:
                st.markdown(f"""
                <div style="padding:8px 10px;background:rgba(22,27,34,0.7);
                    border:1px solid rgba(255,255,255,0.07);border-radius:8px;
                    margin-bottom:6px;font-size:0.82rem;">
                    📄 <strong>{f['filename']}</strong><br>
                    <span style="color:#8892a4;">{f['file_type'].upper()} · {f.get('row_count',0):,} rows · {utils.time_ago(f['uploaded_at'])}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#8892a4;font-size:0.85rem;">No files uploaded yet.</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: UPLOAD DATA █████████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_upload():
    utils.section_header("📂 Upload Data", "Upload CSV, Excel, or PDF files for AI analysis")

    uploaded_files = st.file_uploader(
        "Drag & drop files here or click to browse",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
        key="main_file_uploader",
    )

    if uploaded_files:
        with st.spinner("🔄 Processing files..."):
            all_docs, loaded_results, combined_df = load_multiple_files(uploaded_files)

        success_count = sum(1 for r in loaded_results if not r.get("error"))
        error_count = len(loaded_results) - success_count

        if success_count:
            st.success(f"✅ Successfully loaded {success_count} file(s)!")
        if error_count:
            st.error(f"❌ {error_count} file(s) failed to load.")

        for result in loaded_results:
            with st.expander(f"📄 {result['filename']}", expanded=not result.get("error")):
                if result.get("error"):
                    st.error(result["error"])
                    continue

                col1, col2, col3 = st.columns(3)
                col1.metric("Type", result["type"].upper())
                if result.get("df") is not None:
                    col2.metric("Rows", f"{len(result['df']):,}")
                    col3.metric("Columns", len(result["df"].columns))

                if result.get("df") is not None:
                    st.dataframe(result["df"].head(10), use_container_width=True, hide_index=True)

                # Save metadata to DB
                df_for_meta = result.get("df")
                db.save_file_metadata(
                    user_id=user_id,
                    filename=result["filename"],
                    file_type=result["type"],
                    file_size=0,
                    row_count=len(df_for_meta) if df_for_meta is not None else 0,
                    col_count=len(df_for_meta.columns) if df_for_meta is not None else 0,
                )

        # Build RAG Index
        if all_docs:
            progress = st.progress(0, text="🔄 Starting document indexing…")
            status_box = st.empty()

            status_box.info("📄 Step 1/5 — Loading and parsing documents…")
            progress.progress(10, text="Step 1/5: Parsing documents…")

            status_box.info("✂️ Step 2/5 — Splitting into smart chunks…")
            progress.progress(25, text="Step 2/5: Chunking text…")
            chunks = chunk_documents(
                all_docs,
                chunk_size=settings.get("chunk_size", DEFAULT_CHUNK_SIZE),
                overlap=DEFAULT_CHUNK_OVERLAP
            )

            status_box.info(f"🧠 Step 3/5 — Generating vector embeddings for {len(chunks)} chunks…")
            progress.progress(50, text="Step 3/5: Embedding chunks…")
            embeddings = embed_chunks(chunks, model_name=settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL))

            status_box.info("🗄️ Step 4/5 — Building FAISS vector index…")
            progress.progress(75, text="Step 4/5: Indexing vectors…")
            vs = build_vector_store(chunks, embeddings, user_id=user_id)
            vs.save()

            status_box.info("⚡ Step 5/5 — Building BM25 keyword index…")
            progress.progress(90, text="Step 5/5: Building BM25 index…")
            bm25 = BM25Retriever(chunks)
            st.session_state["bm25_retriever"] = bm25

            progress.progress(100, text="✅ Indexing complete!")
            status_box.empty()

            st.session_state["vector_store"] = vs
            st.session_state["chunks"] = chunks
            st.session_state["file_names"] = [r["filename"] for r in loaded_results]

            st.success(f"🤖 Hybrid RAG index built! {len(chunks)} chunks indexed · BM25 + FAISS ready.")
            st.toast("📚 Documents indexed successfully!", icon="✅")

        if combined_df is not None:
            st.session_state["current_df"] = combined_df
            st.session_state.pop("ai_suggested_charts", None)
            # Load into SQLite for SQL mode
            sql_db_path = utils.load_df_to_sqlite(combined_df, user_id)
            st.session_state["current_sql_db"] = sql_db_path

            st.markdown("---")
            st.markdown("### 🔍 Data Quality Report")
            score, components = analytics.get_data_quality_score(combined_df)
            color = "#3fb950" if score >= 80 else "#f0a800" if score >= 60 else "#f85149"
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Quality Score", f"{score}/100", delta=None)
            sc2.metric("Completeness", f"{components['completeness']}%")
            sc3.metric("Uniqueness", f"{components['uniqueness']}%")
            sc4.metric("Consistency", f"{components['consistency']}%")


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: CHAT WITH AI ████████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_chat():
    utils.section_header("💬 Chat with AI", "Ask anything about your uploaded data")

    vs = st.session_state.get("vector_store")
    df = st.session_state.get("current_df")

    if vs is None and df is None:
        utils.no_data_placeholder("Upload data first to start chatting with the AI!")
        return

    # Mode switcher
    mode_col, k_col = st.columns([3, 1])
    with mode_col:
        chat_mode = st.radio(
            "Chat Mode",
            ["🤖 RAG Q&A", "🗄️ SQL Mode", "🐼 Pandas Mode"],
            horizontal=True,
            key="chat_mode_radio",
        )
    with k_col:
        top_k = st.slider("Top-K", 1, 10, settings.get("top_k", DEFAULT_TOP_K), key="top_k_slider")

    # Show available columns when in SQL / Pandas mode
    if df is not None and ("SQL" in chat_mode or "Pandas" in chat_mode):
        with st.expander("📋 Available Columns (use these exact names in your question)", expanded=False):
            col_df = pd.DataFrame({
                "Column": df.columns.tolist(),
                "Type": [str(df[c].dtype) for c in df.columns],
                "Sample": [str(df[c].iloc[0]) if len(df) > 0 else "" for c in df.columns],
            })
            st.dataframe(col_df, use_container_width=True, hide_index=True)

    # Action buttons
    btn1, btn2, _ = st.columns(3)
    with btn1:
        if st.button("🆕 New Session", use_container_width=True):
            hist.new_session()
            st.rerun()
    with btn2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()

    st.markdown("---")

    # ── Display existing chat messages with feedback buttons ──────────────────
    messages = hist.get_current_messages()
    session_id = hist.get_or_create_session_id()

    for msg_idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(format_sources(msg["sources"]))

        # Feedback buttons only on assistant messages
        if msg["role"] == "assistant":
            fb_key_up   = f"fb_up_{session_id}_{msg_idx}"
            fb_key_down = f"fb_down_{session_id}_{msg_idx}"
            fb_state_key = f"fb_state_{session_id}_{msg_idx}"

            # Show feedback row only if not already rated
            if not st.session_state.get(fb_state_key):
                fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 8])
                with fb_c1:
                    if st.button("👍", key=fb_key_up, help="Helpful answer"):
                        db.save_feedback(user_id, session_id, msg_idx, "up")
                        st.session_state[fb_state_key] = "up"
                        st.toast("Thanks for your feedback! 👍", icon="✅")
                        st.rerun()
                with fb_c2:
                    if st.button("👎", key=fb_key_down, help="Not helpful"):
                        db.save_feedback(user_id, session_id, msg_idx, "down")
                        st.session_state[fb_state_key] = "down"
                        st.toast("Thanks! We'll work to improve. 👎", icon="📝")
                        st.rerun()
            else:
                rating = st.session_state[fb_state_key]
                icon = "👍" if rating == "up" else "👎"
                st.caption(f"{icon} Feedback recorded")

    # ── Chat input ────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask anything about your data… (follow-up questions work too!)", key="chat_input")
    if not prompt:
        return

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    hist.save_message(user_id, "user", prompt)

    model       = settings.get("model", DEFAULT_MODEL)
    temperature = settings.get("temperature", DEFAULT_TEMPERATURE)
    eff_api_key = st.session_state.get("groq_api_key") or api_key

    # Build conversation history for memory (last 8 messages = 4 exchanges)
    conversation_history = hist.get_current_messages()[-8:]

    with st.chat_message("assistant"):

        # ── RAG Mode (with Hybrid Search + Streaming + Web Fallback) ──────────
        if "RAG" in chat_mode:
            if vs is None:
                answer = "⚠️ No RAG index found. Please upload and index documents first."
                sources = []
                st.markdown(answer)
            else:
                bm25 = st.session_state.get("bm25_retriever")

                # Rebuild BM25 index if missing (e.g. after app restart)
                if bm25 is None and vs.get_all_chunks():
                    bm25 = BM25Retriever(vs.get_all_chunks())
                    st.session_state["bm25_retriever"] = bm25

                context, sources, is_confident = retrieve_with_context(
                    prompt, vs, k=top_k, bm25_retriever=bm25
                )

                use_web = False
                if not context or not is_confident:
                    # Fall back to web search
                    web_results = web_search(prompt, max_results=5)
                    if web_results:
                        use_web = True
                        context = format_web_results_as_context(web_results)
                        sources = []  # web sources shown separately
                        st.info("📄 No relevant content found in your documents. Using web search instead…")

                if not context:
                    answer = "I couldn't find relevant information in the uploaded documents or via web search. Please try rephrasing your question."
                    sources = []
                    st.markdown(answer)
                else:
                    # ── Stream the answer ──────────────────────────────────
                    stream_gen = llm_module.ask_llm_stream(
                        question=prompt,
                        context=context,
                        model=model,
                        temperature=temperature,
                        api_key=eff_api_key,
                        conversation_history=conversation_history,
                        use_web_context=use_web,
                    )
                    answer = st.write_stream(stream_gen)

                    # Show sources
                    if use_web:
                        st.markdown(format_web_sources(web_results))
                    elif sources:
                        st.markdown(format_sources(sources))

        # ── SQL Mode ──────────────────────────────────────────────────────────
        elif "SQL" in chat_mode:
            sources = []
            if df is None:
                answer = "⚠️ No tabular data loaded. Please upload a CSV or Excel file first."
                st.markdown(answer)
            else:
                with st.spinner("Generating SQL…"):
                    cols = df.columns.tolist()
                    if not st.session_state.get("current_sql_db"):
                        sql_db_path = utils.load_df_to_sqlite(df, user_id)
                        st.session_state["current_sql_db"] = sql_db_path

                    dtype_lines = "\n".join(f"  - {c} ({str(df[c].dtype)})" for c in cols)
                    sample      = df.head(5).to_string(index=False)
                    schema_hint = (
                        f"Table: data_table\n"
                        f"Exact column names (use these EXACTLY, do not rename):\n{dtype_lines}\n\n"
                        f"Sample rows:\n{sample}"
                    )

                    llm_response = llm_module.generate_sql(
                        prompt, cols, schema_hint, model=model, api_key=eff_api_key
                    )
                    sql_code = llm_module.extract_sql(llm_response)

                    if sql_code:
                        result_df, err = utils.execute_sql_query(
                            st.session_state["current_sql_db"], sql_code
                        )
                        if err:
                            # Auto-retry
                            retry_prompt = (
                                f"The previous query failed with: {err}\n\n"
                                f"The exact available columns are: {cols}\n"
                                f"Please fix the SQL query. Use ONLY these column names."
                            )
                            retry_response = llm_module.generate_sql(
                                retry_prompt, cols, schema_hint, model=model, api_key=eff_api_key
                            )
                            retry_sql = llm_module.extract_sql(retry_response)
                            if retry_sql:
                                result_df, err2 = utils.execute_sql_query(
                                    st.session_state["current_sql_db"], retry_sql
                                )
                                if err2:
                                    answer = (
                                        f"**SQL Response:**\n\n{retry_response}\n\n"
                                        f"⚠️ Execution error: {err2}\n\n"
                                        f"**Available columns:** `{'`, `'.join(cols)}`"
                                    )
                                    st.markdown(answer)
                                else:
                                    answer = f"**SQL Response (auto-corrected):**\n\n{retry_response}"
                                    st.markdown(answer)
                                    if result_df is not None and not result_df.empty:
                                        st.dataframe(result_df, use_container_width=True, key="sql_retry_result")
                                        answer += f"\n\n*Query returned {len(result_df)} row(s).*"
                                    hist.save_message(user_id, "assistant", answer, sources)
                                    return
                            else:
                                answer = (
                                    f"**SQL Response:**\n\n{llm_response}\n\n"
                                    f"⚠️ Execution error: {err}\n\n"
                                    f"**Available columns:** `{'`, `'.join(cols)}`"
                                )
                                st.markdown(answer)
                        else:
                            answer = f"**SQL Response:**\n\n{llm_response}"
                            st.markdown(answer)
                            if result_df is not None and not result_df.empty:
                                st.dataframe(result_df, use_container_width=True, key="sql_result_df")
                                answer += f"\n\n*Query returned {len(result_df)} row(s).*"
                            hist.save_message(user_id, "assistant", answer, sources)
                            return
                    else:
                        answer = (
                            f"{llm_response}\n\n"
                            f"**Available columns:** `{'`, `'.join(cols)}`"
                        )
                        st.markdown(answer)

        # ── Pandas Mode ───────────────────────────────────────────────────────
        elif "Pandas" in chat_mode:
            sources = []
            if df is None:
                answer = "⚠️ No tabular data loaded. Please upload a CSV or Excel file first."
                st.markdown(answer)
            else:
                with st.spinner("Generating pandas code…"):
                    cols   = df.columns.tolist()
                    dtypes = df.dtypes.to_string()
                    sample = df.head(5).to_string(index=False)
                    llm_response = llm_module.generate_pandas_code(
                        prompt, cols, dtypes, sample, model=model, api_key=eff_api_key
                    )
                    py_code = llm_module.extract_python_code(llm_response)

                    if py_code:
                        exec_result, exec_err = utils.execute_pandas_code(py_code, df)
                        answer = f"**Analysis:**\n\n{llm_response}"
                        st.markdown(answer)
                        if exec_err:
                            st.error(f"Execution error: {exec_err}")
                        elif exec_result is not None:
                            if isinstance(exec_result, pd.DataFrame):
                                st.dataframe(exec_result, use_container_width=True, key="pandas_result_df")
                            elif isinstance(exec_result, pd.Series):
                                st.dataframe(exec_result.reset_index(), use_container_width=True, key="pandas_series_df")
                            else:
                                st.metric("Result", str(exec_result))
                        hist.save_message(user_id, "assistant", answer, sources)
                        return
                    else:
                        answer = llm_response
                        st.markdown(answer)

    hist.save_message(user_id, "assistant", answer, sources if "sources" in locals() else [])


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: ANALYTICS DASHBOARD █████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_analytics():
    utils.section_header("📊 Analytics Dashboard", "Explore, clean, and visualize your data")

    df = st.session_state.get("current_df")
    if df is None:
        utils.no_data_placeholder()
        return

    tab_eda, tab_clean, tab_charts = st.tabs(["🔍 EDA Report", "🧹 Data Cleaning", "📈 Charts"])

    # ── EDA Tab ───────────────────────────────────────────────────────────────
    with tab_eda:
        info = analytics.get_basic_info(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{info['rows']:,}")
        c2.metric("Columns", info["columns"])
        c3.metric("Memory", f"{info['memory_mb']} MB")
        c4.metric("Numeric Cols", len(info["numeric_columns"]))

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📋 Summary Statistics")
            stats = analytics.get_summary_stats(df)
            if not stats.empty:
                st.dataframe(stats, use_container_width=True)
            else:
                st.info("No numeric columns to summarize.")

            st.markdown("#### ❓ Missing Values")
            missing = analytics.get_missing_value_report(df)
            if missing.empty:
                utils.success_box("No missing values found! 🎉")
            else:
                st.dataframe(missing, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("#### 📊 Data Types")
            dtype_df = pd.DataFrame({"Column": info["column_names"],
                                     "Type": [str(df[c].dtype) for c in info["column_names"]]})
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

            st.markdown("#### 🔢 Unique Values")
            unique_report = analytics.get_unique_value_report(df)
            st.dataframe(unique_report[["Column", "Unique Values", "Uniqueness %"]], use_container_width=True, hide_index=True)

        st.markdown("#### 🎯 Outlier Detection (IQR Method)")
        outliers = analytics.detect_outliers(df)
        if outliers.empty:
            utils.info_box("No outliers detected in numeric columns.")
        else:
            st.dataframe(outliers, use_container_width=True, hide_index=True)

        st.markdown("#### 🔗 Correlation Matrix")
        corr_fig = ch.heatmap_chart(df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True, key="eda_corr_heatmap")
        else:
            st.info("Need at least 2 numeric columns for correlation matrix.")

    # ── Data Cleaning Tab ─────────────────────────────────────────────────────
    with tab_clean:
        dup = analytics.get_duplicate_report(df)
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Rows", f"{dup['total_rows']:,}")
        d2.metric("Duplicate Rows", f"{dup['duplicate_rows']:,}")
        d3.metric("Unique Rows", f"{dup['unique_rows']:,}")

        with st.expander("🗂️ Remove Duplicates"):
            if st.button("Remove Duplicate Rows", type="primary"):
                cleaned, n = analytics.remove_duplicates(df)
                st.session_state["current_df"] = cleaned
                st.success(f"✅ Removed {n} duplicate row(s).")
                st.rerun()

        with st.expander("🔧 Fill Missing Values"):
            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("Column", ["All Columns"] + df.columns.tolist(), key="fill_col")
            with col2:
                strategy = st.selectbox("Strategy", ["mean", "median", "mode", "zero", "ffill", "bfill"], key="fill_strategy")
            if st.button("Apply Fill Strategy", type="primary"):
                col_arg = None if target_col == "All Columns" else target_col
                cleaned = analytics.fill_missing_values(df, strategy=strategy, column=col_arg)
                st.session_state["current_df"] = cleaned
                st.success("✅ Missing values filled.")
                st.rerun()

        with st.expander("🗑️ Drop Missing Rows"):
            col1, col2 = st.columns(2)
            with col1:
                drop_col = st.selectbox("Drop where column is null", ["Any Column"] + df.columns.tolist(), key="drop_col")
            with col2:
                if st.button("Drop Rows", type="primary"):
                    col_arg = None if drop_col == "Any Column" else drop_col
                    cleaned, n = analytics.drop_missing_rows(df, column=col_arg)
                    st.session_state["current_df"] = cleaned
                    st.success(f"✅ Dropped {n} row(s) with null values.")
                    st.rerun()

        with st.expander("🗑️ Drop Column"):
            col_to_drop = st.selectbox("Column to drop", df.columns.tolist(), key="col_drop")
            if st.button("Drop Column", type="primary"):
                cleaned = analytics.drop_column(df, col_to_drop)
                st.session_state["current_df"] = cleaned
                st.success(f"✅ Dropped column '{col_to_drop}'.")
                st.rerun()

        with st.expander("✏️ Rename Column"):
            col1, col2 = st.columns(2)
            with col1:
                old_name = st.selectbox("Old Name", df.columns.tolist(), key="old_col_name")
            with col2:
                new_name = st.text_input("New Name", key="new_col_name")
            if st.button("Rename Column", type="primary") and new_name:
                cleaned = analytics.rename_column(df, old_name, new_name)
                st.session_state["current_df"] = cleaned
                st.success(f"✅ Renamed '{old_name}' → '{new_name}'.")
                st.rerun()

        with st.expander("🔄 Change Data Type"):
            col1, col2 = st.columns(2)
            with col1:
                type_col = st.selectbox("Column", df.columns.tolist(), key="type_col")
            with col2:
                target_type = st.selectbox("Target Type", ["int", "float", "str", "datetime", "bool"], key="target_type")
            if st.button("Change Type", type="primary"):
                cleaned, err = analytics.change_dtype(df, type_col, target_type)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.session_state["current_df"] = cleaned
                    st.success(f"✅ Changed '{type_col}' to {target_type}.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### 📥 Export Cleaned Data")
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("⬇️ Download as CSV", data=export_csv(df), file_name="cleaned_data.csv",
                               mime="text/csv", use_container_width=True)
        with e2:
            st.download_button("⬇️ Download as Excel", data=export_excel(df), file_name="cleaned_data.xlsx",
                               mime="application/vnd.ms-excel", use_container_width=True)

    # ── Charts Tab ────────────────────────────────────────────────────────────
    with tab_charts:
        st.markdown("### 🤖 Auto-Generated Charts")
        
        # Automatic cache invalidation if columns changed
        if "ai_suggested_charts" in st.session_state:
            current_cols = set(df.columns)
            cache_valid = True
            for s in st.session_state["ai_suggested_charts"]:
                params = s.get("params", {})
                for k, col in params.items():
                    if isinstance(col, str) and col and col not in current_cols:
                        cache_valid = False
                        break
                    elif isinstance(col, list):
                        for c in col:
                            if c not in current_cols:
                                cache_valid = False
                                break
                if not cache_valid:
                    break
            if not cache_valid:
                st.session_state.pop("ai_suggested_charts", None)

        if "ai_suggested_charts" not in st.session_state:
            with st.spinner("🧠 AI is analyzing dataset columns to suggest valuable charts..."):
                cols = df.columns.tolist()
                dtypes = df.dtypes.to_string()
                sample = df.head(5).to_string(index=False)
                eff_api_key = st.session_state.get("groq_api_key") or api_key
                model = settings.get("model", DEFAULT_MODEL)
                
                ai_suggestions = []
                if eff_api_key:
                    try:
                        ai_suggestions = llm_module.suggest_charts_with_ai(
                            columns=cols,
                            dtypes=dtypes,
                            sample_data=sample,
                            model=model,
                            api_key=eff_api_key
                        )
                    except Exception:
                        pass
                
                # Fallback to heuristics if AI failed or key missing
                if not ai_suggestions:
                    ai_suggestions = ch.auto_suggest_charts(df)
                
                st.session_state["ai_suggested_charts"] = ai_suggestions

        suggestions = st.session_state["ai_suggested_charts"]

        if suggestions:
            for i in range(0, len(suggestions), 2):
                grid_cols = st.columns(2)
                for j, grid_col in enumerate(grid_cols):
                    if i + j < len(suggestions):
                        s = suggestions[i + j]
                        fig = ch.build_chart(df, s["type"], s["params"], s["title"])
                        if fig:
                            grid_col.plotly_chart(
                                fig, use_container_width=True,
                                key=f"auto_chart_{i}_{j}_{s['type']}"
                            )

        st.markdown("---")
        st.markdown("### 🎨 Custom Chart Builder")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        all_cols = df.columns.tolist()

        cb1, cb2 = st.columns(2)
        with cb1:
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart", "Histogram",
                                                      "Scatter Plot", "Area Chart", "Box Plot"], key="custom_chart_type")
        with cb2:
            custom_title = st.text_input("Chart Title", key="custom_chart_title", placeholder="My Custom Chart")

        p1, p2, p3 = st.columns(3)
        with p1:
            x_col = st.selectbox("X Axis", [""] + all_cols, key="chart_x")
        with p2:
            y_col = st.selectbox("Y Axis", [""] + num_cols, key="chart_y")
        with p3:
            color_col = st.selectbox("Color By (optional)", ["None"] + cat_cols, key="chart_color")

        if st.button("🎨 Generate Chart", type="primary"):
            params = {}
            if x_col:
                params["x"] = x_col
            if y_col:
                params["y"] = y_col
            if color_col and color_col != "None":
                params["color"] = color_col

            fig = ch.build_chart(df, chart_type, params, custom_title or chart_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="custom_chart_output")
            else:
                st.error("Could not generate chart with the selected parameters.")


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: BUSINESS INSIGHTS ███████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_insights():
    utils.section_header("💡 Business Insights", "AI-powered analysis of your data")

    df = st.session_state.get("current_df")
    if df is None:
        utils.no_data_placeholder()
        return

    eff_api_key = st.session_state.get("groq_api_key") or api_key

    # KPI Cards
    st.markdown("### 📊 Key Performance Indicators")
    kpis = ins.extract_kpis(df)
    kpi_cols = st.columns(min(4, len(kpis)))
    for i, kpi in enumerate(kpis[:4]):
        with kpi_cols[i % 4]:
            formatted = ins.format_kpi_value(kpi["value"], kpi["format"])
            utils.metric_card(kpi["label"], formatted, kpi["icon"], "#667eea")

    if len(kpis) > 4:
        kpi_cols2 = st.columns(min(4, len(kpis) - 4))
        for i, kpi in enumerate(kpis[4:8]):
            with kpi_cols2[i % 4]:
                formatted = ins.format_kpi_value(kpi["value"], kpi["format"])
                utils.metric_card(kpi["label"], formatted, kpi["icon"], "#764ba2")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🏆 Business Insight Cards")
        auto_insights = ins.generate_auto_insights(df)
        for insight in auto_insights:
            utils.insight_card(
                insight["title"], insight["value"],
                insight["subtitle"], insight["icon"], insight["color"]
            )

    with col_right:
        st.markdown("### 📈 Top Performers")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if cat_cols and num_cols:
            tp1, tp2 = st.columns(2)
            with tp1:
                group_col = st.selectbox("Group By", cat_cols, key="top_group")
            with tp2:
                val_col = st.selectbox("Value", num_cols, key="top_val")

            top10 = ins.get_top_performers(df, group_col, val_col, n=10)
            if not top10.empty:
                fig = ch.bar_chart(top10, x=group_col, y=val_col, title=f"Top 10 {group_col} by {val_col}")
                st.plotly_chart(fig, use_container_width=True, key="insights_top_performers")

    # Trend Analysis
    date_col = ins.detect_date_column(df)
    if date_col and num_cols:
        st.markdown("---")
        st.markdown("### 📅 Trend Analysis")
        trend_col = st.selectbox("Metric", num_cols, key="trend_metric")
        trend_tab1, trend_tab2, trend_tab3 = st.tabs(["Monthly", "Quarterly", "Yearly"])

        with trend_tab1:
            monthly = ins.get_monthly_trend(df, date_col, trend_col)
            if not monthly.empty:
                growth = ins.calculate_growth(monthly, trend_col)
                st.metric("Overall Monthly Growth", f"{growth:+.1f}%")
                fig = ch.area_chart(monthly, x="Month", y=trend_col, title=f"Monthly {trend_col}")
                st.plotly_chart(fig, use_container_width=True, key=f"trend_monthly_{trend_col}")

        with trend_tab2:
            quarterly = ins.get_quarterly_trend(df, date_col, trend_col)
            if not quarterly.empty:
                growth = ins.calculate_growth(quarterly, trend_col)
                st.metric("Overall Quarterly Growth", f"{growth:+.1f}%")
                fig = ch.bar_chart(quarterly, x="Quarter", y=trend_col, title=f"Quarterly {trend_col}")
                st.plotly_chart(fig, use_container_width=True, key=f"trend_quarterly_{trend_col}")

        with trend_tab3:
            yearly = ins.get_yearly_trend(df, date_col, trend_col)
            if not yearly.empty:
                growth = ins.calculate_growth(yearly, trend_col)
                st.metric("Overall Yearly Growth", f"{growth:+.1f}%")
                fig = ch.line_chart(yearly, x="Year", y=trend_col, title=f"Yearly {trend_col}")
                st.plotly_chart(fig, use_container_width=True, key=f"trend_yearly_{trend_col}")

    # AI Recommendations
    st.markdown("---")
    st.markdown("### 🧠 AI-Generated Business Recommendations")
    if st.button("✨ Generate AI Insights", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and generating insights…"):
            summary = analytics.generate_eda_summary_text(df)
            ai_insights = llm_module.generate_insights(summary, model=settings.get("model", DEFAULT_MODEL),
                                                        api_key=eff_api_key)
        st.markdown(ai_insights)


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: REPORTS █████████████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_reports():
    utils.section_header("📋 Reports", "Generate and export professional AI reports")

    df = st.session_state.get("current_df")
    eff_api_key = st.session_state.get("groq_api_key") or api_key

    tab_gen, tab_hist = st.tabs(["📄 Generate Report", "📁 Report History"])

    with tab_gen:
        if df is None:
            utils.no_data_placeholder()
            return

        with st.form("generate_report_form"):
            st.markdown("### 📝 Report Configuration")
            report_title = st.text_input("Report Title", value=f"Analysis Report — {datetime.now().strftime('%B %d, %Y')}")
            include_ai_summary = st.checkbox("Include AI Executive Summary", value=True)
            include_kpis = st.checkbox("Include KPI Dashboard", value=True)
            include_insights = st.checkbox("Include Business Insights", value=True)
            submitted = st.form_submit_button("🚀 Generate PDF Report", type="primary", use_container_width=True)

        if submitted:
            with st.spinner("🔄 Generating professional report..."):
                ai_summary = ""
                if include_ai_summary:
                    summary_text = analytics.generate_eda_summary_text(df)
                    ai_summary = llm_module.generate_insights(summary_text, model=settings.get("model", DEFAULT_MODEL),
                                                              api_key=eff_api_key)
                kpis = ins.extract_kpis(df) if include_kpis else []
                insights_list = ins.generate_auto_insights(df) if include_insights else []

                pdf_bytes, file_path = generate_pdf_report(
                    title=report_title, df=df, insights=insights_list,
                    ai_summary=ai_summary, kpis=[
                        {"icon": k["icon"], "label": k["label"],
                         "value": ins.format_kpi_value(k["value"], k["format"])}
                        for k in kpis
                    ],
                    username=user.get("full_name") or user.get("username", "User"),
                    user_id=user_id,
                )
                db.save_report(user_id, report_title, file_path, "PDF")

            st.success("✅ Report generated successfully!")
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"{report_title.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

        st.markdown("---")
        st.markdown("### 📥 Quick Data Export")
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("⬇️ Export CSV", data=export_csv(df), file_name="data_export.csv",
                               mime="text/csv", use_container_width=True)
        with e2:
            st.download_button("⬇️ Export Excel", data=export_excel(df), file_name="data_export.xlsx",
                               mime="application/vnd.ms-excel", use_container_width=True)

    with tab_hist:
        reports = db.get_user_reports(user_id)
        if not reports:
            st.info("No reports generated yet.")
        else:
            for r in reports:
                with st.expander(f"📄 {r['title']} — {utils.time_ago(r['created_at'])}"):
                    st.write(f"**Type:** {r['report_type']}")
                    st.write(f"**Generated:** {r['created_at']}")
                    if r.get("file_path") and Path(r["file_path"]).exists():
                        with open(r["file_path"], "rb") as f:
                            st.download_button("⬇️ Download", data=f.read(),
                                               file_name=Path(r["file_path"]).name,
                                               mime="application/pdf", key=f"dl_{r['id']}")


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: CHAT HISTORY ████████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_history():
    utils.section_header("🕐 Chat History", "Browse, search, and manage your conversations")

    tab_browse, tab_search = st.tabs(["📋 Browse Sessions", "🔍 Search History"])

    with tab_browse:
        sessions = hist.get_all_sessions(user_id)
        if not sessions:
            st.info("No chat history found.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Delete All History", type="secondary"):
                    hist.clear_all_history(user_id)
                    st.success("All history deleted.")
                    st.rerun()

            for s in sessions:
                with st.expander(f"🗣️ Session — {utils.time_ago(s['started_at'])} ({s['message_count']} messages)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("▶️ Continue Session", key=f"continue_{s['session_id']}"):
                            hist.load_session(user_id, s["session_id"])
                            st.session_state["current_page"] = "chat"
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{s['session_id']}"):
                            hist.delete_session(user_id, s["session_id"])
                            st.rerun()

                    # Preview messages
                    msgs = db.get_chat_history(user_id, s["session_id"])
                    for m in msgs[:6]:
                        role_icon = "👤" if m["role"] == "user" else "🤖"
                        bg = "rgba(0,92,75,0.2)" if m["role"] == "user" else "rgba(22,27,34,0.7)"
                        st.markdown(
                            f'<div style="background:{bg};border-radius:8px;padding:8px 12px;'
                            f'margin:4px 0;font-size:0.85rem;">'
                            f'{role_icon} {utils.truncate_text(m["content"], 150)}</div>',
                            unsafe_allow_html=True
                        )

    with tab_search:
        query = st.text_input("🔍 Search conversations…", key="history_search_query")
        if query:
            results = hist.search_history(user_id, query)
            if not results:
                st.info("No results found.")
            else:
                st.success(f"Found {len(results)} matching message(s).")
                for m in results:
                    role_icon = "👤" if m["role"] == "user" else "🤖"
                    st.markdown(f"""
                    <div style="background:rgba(22,27,34,0.7);border:1px solid rgba(102,126,234,0.2);
                        border-radius:10px;padding:12px 16px;margin-bottom:8px;">
                        <div style="color:#8892a4;font-size:0.75rem;margin-bottom:4px;">
                            {role_icon} {m['role'].capitalize()} · {utils.time_ago(m['created_at'])}
                        </div>
                        <div style="color:#e6edf3;">{utils.truncate_text(m['content'], 300)}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: SETTINGS ████████████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_settings():
    utils.section_header("⚙️ Settings", "Configure your AI Data Analyst experience")

    current = db.get_user_settings(user_id)

    with st.form("settings_form"):
        st.markdown("#### 🤖 AI Model Settings")
        c1, c2 = st.columns(2)
        with c1:
            model = st.selectbox("LLM Model", AVAILABLE_MODELS,
                                 index=AVAILABLE_MODELS.index(current.get("model", DEFAULT_MODEL))
                                 if current.get("model", DEFAULT_MODEL) in AVAILABLE_MODELS else 0)
            temperature = st.slider("Temperature", 0.0, 1.0,
                                    float(current.get("temperature", DEFAULT_TEMPERATURE)), step=0.05,
                                    help="Higher = more creative, Lower = more precise")
        with c2:
            top_k = st.slider("Top-K Retrieval", 1, 15, int(current.get("top_k", DEFAULT_TOP_K)),
                               help="Number of document chunks to retrieve per query")
            chunk_size = st.slider("Chunk Size", 200, 1500, int(current.get("chunk_size", DEFAULT_CHUNK_SIZE)),
                                   step=50, help="Size of each text chunk for RAG indexing")

        st.markdown("#### 🧠 Embedding Settings")
        embedding_model = st.selectbox("Embedding Model", AVAILABLE_EMBEDDING_MODELS,
                                        index=AVAILABLE_EMBEDDING_MODELS.index(
                                            current.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
                                        if current.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
                                        in AVAILABLE_EMBEDDING_MODELS else 0)

        st.markdown("#### 🔑 API Key")
        api_key_input = st.text_input("Groq API Key", type="password",
                                       value=st.session_state.get("groq_api_key", ""),
                                       placeholder="Enter your Groq API key (gsk_...)")

        submitted = st.form_submit_button("💾 Save Settings", type="primary", use_container_width=True)
        if submitted:
            db.update_user_settings(user_id,
                model=model, temperature=temperature, top_k=top_k,
                chunk_size=chunk_size, embedding_model=embedding_model)
            if api_key_input:
                st.session_state["groq_api_key"] = api_key_input
            st.session_state["settings"] = db.get_user_settings(user_id)
            st.success("✅ Settings saved!")

    st.markdown("---")
    st.markdown("#### 👤 Account Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Username:** `{user.get('username')}`")
        st.markdown(f"**Email:** `{user.get('email')}`")
    with col2:
        st.markdown(f"**Member Since:** {user.get('created_at', 'N/A')[:10]}")
        st.markdown(f"**Last Login:** {user.get('last_login', 'N/A')}")

    st.markdown("---")
    st.markdown("#### 🗑️ Danger Zone")
    if st.button("🗑️ Clear All Chat History", type="secondary"):
        hist.clear_all_history(user_id)
        st.success("All chat history cleared.")
    if st.button("🔄 Reset Index", type="secondary"):
        st.session_state["vector_store"] = None
        st.session_state["chunks"] = None
        st.session_state["current_df"] = None
        st.session_state["current_sql_db"] = None
        st.session_state["chat_messages"] = []
        st.success("Session data cleared. Please re-upload your files.")


# ──────────────────────────────────────────────────────────────────────────────
# ██████████████████████ PAGE: DEVELOPER ███████████████████████████████████████
# ──────────────────────────────────────────────────────────────────────────────
def page_developer():
    utils.section_header("👨‍💻 Developer Profile", "Learn more about the creator of AI Data Analyst Assistant")

    col_profile, col_form = st.columns([1, 1])

    with col_profile:
        # Glassmorphic Profile Card
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
            border-radius:15px;padding:24px;text-align:center;box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
            <div style="font-size:5rem;margin-bottom:12px;">👨‍💻</div>
            <h2 style="margin:0;color:#e6edf3;font-size:1.8rem;">Gopi Chand Pasam</h2>
            <p style="color:#667eea;font-weight:600;margin:4px 0 16px 0;font-size:1.1rem;">Full Stack AI & RAG Engineer</p>
            <hr style="border:0;border-top:1px solid rgba(255,255,255,0.1);margin:16px 0;">
            <div style="text-align:left;color:#e6edf3;font-size:0.9rem;">
                <p>🚀 <strong>Specialization:</strong> Building highly optimized Retrieval-Augmented Generation (RAG) pipelines, analytical assistants, and secure enterprise web apps.</p>
                <p>🛠️ <strong>Core Stack:</strong> Python · Streamlit · Groq API · FAISS · BM25 · SQLite · Pytest · pandas · Plotly</p>
                <p>📬 <strong>Email:</strong> <a href="mailto:gopipasam93@gmail.com" style="color:#667eea;text-decoration:none;">gopipasam93@gmail.com</a></p>
            </div>
            <div style="display:flex;justify-content:center;gap:15px;margin-top:20px;">
                <a href="https://github.com/gopi463" target="_blank" style="text-decoration:none;
                    background:#24292e;color:white;padding:8px 16px;border-radius:8px;font-weight:600;font-size:0.85rem;">
                    🐙 GitHub Profile
                </a>
                <a href="https://linkedin.com" target="_blank" style="text-decoration:none;
                    background:#0a66c2;color:white;padding:8px 16px;border-radius:8px;font-weight:600;font-size:0.85rem;">
                    💼 LinkedIn Profile
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### 🌟 About the Project")
        st.markdown(
            "This project is a next-generation **AI Data Analyst Assistant** designed to bridge the gap between "
            "raw spreadsheets and enterprise-level business intelligence. By combining vector similarity embeddings (FAISS) "
            "with exact lexical keywords retrieval (BM25Okapi) and reciprocal rank fusion (RRF) reranking, "
            "it delivers extreme precision for document Q&A alongside automated SQL/Pandas code generation."
        )

    with col_form:
        st.markdown("### ✉️ Get in Touch / Hire Me")
        st.markdown("Are you a recruiter or looking for a freelancer? Send an inquiry directly through this app form!")

        with st.form("recruiter_form"):
            name = st.text_input("Name *", placeholder="Enter your name")
            company = st.text_input("Company", placeholder="Enter your company")
            email = st.text_input("Email *", placeholder="Enter your contact email")
            message = st.text_area("Message *", placeholder="Briefly describe the role, project, or why you are reaching out...")
            
            submitted = st.form_submit_button("🚀 Send Message", type="primary", use_container_width=True)
            if submitted:
                if not name or not email or not message:
                    st.error("⚠️ Please fill in all required fields (*) before sending.")
                else:
                    db.save_inquiry(name, company, email, message)
                    st.success("✅ Inquiry sent successfully! The developer has been notified.")
                    st.toast("Message sent!", icon="✉️")

        # Inquiries Panel (visible in the app for review/demo)
        inquiries = db.get_inquiries()
        if inquiries:
            with st.expander(f"📥 Incoming Inquiries Dashboard ({len(inquiries)})", expanded=False):
                st.info("💡 As a developer, all recruiter messages sent through the contact form are stored in the SQLite DB and listed here for easy follow-up.")
                for inq in inquiries:
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:10px;padding:12px;margin-bottom:10px;">
                        <div style="display:flex;justify-content:between;font-size:0.75rem;color:#8892a4;margin-bottom:4px;">
                            <strong>From: {inq['name']} ({inq['company'] or 'Independent'})</strong>
                            <span style="margin-left:auto;">{inq['created_at'][:16]}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#c8d4ff;margin-bottom:4px;">📧 {inq['email']}</div>
                        <div style="font-size:0.85rem;color:#e6edf3;white-space:pre-wrap;">{inq['message']}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────
PAGE_ROUTER = {
    "home":      page_home,
    "upload":    page_upload,
    "chat":      page_chat,
    "analytics": page_analytics,
    "insights":  page_insights,
    "reports":   page_reports,
    "history":   page_history,
    "settings":  page_settings,
    "developer": page_developer,
}

page_fn = PAGE_ROUTER.get(current_page, page_home)
try:
    page_fn()
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred: {str(e)}")
    st.info("Please try again or contact support if the issue persists.")
    if st.button("🔄 Reload Page"):
        st.rerun()