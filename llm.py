"""
llm.py — Groq LLM Client
AI Data Analyst Assistant

Provides: RAG Q&A (with streaming + conversation memory), SQL generation,
Pandas code generation, business insights.
"""

import os
import re
import json
from typing import Optional, List, Iterator
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# Client Factory
# ──────────────────────────────────────────
def get_groq_client(api_key: Optional[str] = None) -> Optional[Groq]:
    """Lazily create a Groq client. Returns None if no API key available."""
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not key:
        return None
    return Groq(api_key=key)


# ──────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an expert AI Data Analyst Assistant.

You have been provided with context extracted from the user's uploaded documents.
You also have access to the conversation history — use it to handle follow-up questions naturally.

Rules:
1. Answer based on the provided context. You may synthesize, evaluate, and reason from it.
2. If the answer is numeric, be precise and include units.
3. If comparing, structure the comparison clearly.
4. If context is insufficient, say: "I couldn't find enough information in the uploaded documents to answer this."
5. Never hallucinate facts not in the context.
6. Use the conversation history to resolve pronouns and follow-up references (e.g. "it", "that", "the second one").
7. Be concise, professional, and insightful.
"""

WEB_RAG_SYSTEM_PROMPT = """You are an expert AI Data Analyst Assistant.

The user's question could not be answered from the uploaded documents.
You have been provided with real-time web search results as context instead.

Rules:
1. Answer based ONLY on the provided web search results.
2. Cite your sources where possible.
3. Be clear that this information comes from the web, not the user's uploaded documents.
4. Be concise and factual.
"""

SQL_SYSTEM_PROMPT = """You are an expert SQL analyst. The user has a dataset loaded into a SQLite database table called `data_table`.

Rules:
1. Generate valid SQLite SQL queries ONLY.
2. Wrap the SQL code in ```sql ... ``` code blocks.
3. After the SQL, briefly explain what it does.
4. Avoid destructive operations (DROP, DELETE, UPDATE, INSERT).
5. If the request is unclear, ask for clarification.
"""

PANDAS_SYSTEM_PROMPT = """You are an expert Python/pandas data analyst.

The user has a pandas DataFrame available as `df`. Generate safe pandas code to answer the question.

Rules:
1. Wrap ALL code in ```python ... ``` code blocks.
2. Assign the final result to a variable called `result`.
3. Do NOT use file I/O, network calls, or os operations.
4. Be concise and correct.
5. After the code, briefly explain what it does.
"""

INSIGHTS_SYSTEM_PROMPT = """You are an expert business analyst. Analyze the provided data summary and generate actionable business insights.

Format your response with clear sections:
- **Executive Summary**: 2-3 sentence overview
- **Key Findings**: Bulleted list of top findings with numbers
- **Trends**: Notable patterns
- **Recommendations**: 3-5 concrete action items
- **Risk Factors**: Any concerns or anomalies

Be specific. Include percentages and actual values from the data.
"""


# ──────────────────────────────────────────
# Conversation History Formatter
# ──────────────────────────────────────────
def _build_message_history(
    conversation_history: Optional[List[dict]],
    max_exchanges: int = 4,
) -> List[dict]:
    """
    Convert session chat_messages into Groq-compatible message dicts.
    Takes only the last `max_exchanges` user+assistant pairs (= 2*max_exchanges messages).
    Strips 'sources' and other non-standard fields.
    """
    if not conversation_history:
        return []

    # Filter to user/assistant only and take tail
    valid = [
        m for m in conversation_history
        if m.get("role") in ("user", "assistant")
    ]
    tail = valid[-(max_exchanges * 2):]

    return [
        {"role": m["role"], "content": str(m.get("content", ""))[:2000]}
        for m in tail
    ]


# ──────────────────────────────────────────
# Core LLM Call (Non-Streaming)
# ──────────────────────────────────────────
def _call_llm(
    system_prompt: str,
    user_message: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """Make a single LLM call and return the response text."""
    client = get_groq_client(api_key)
    if not client:
        return "⚠️ Error: Groq API key is not configured. Please add it in Settings or the sidebar."

    history_msgs = _build_message_history(conversation_history)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    answer = response.choices[0].message.content or ""
    # Strip any think tags from reasoning models
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer


# ──────────────────────────────────────────
# Streaming LLM Call
# ──────────────────────────────────────────
def _call_llm_stream(
    system_prompt: str,
    user_message: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    conversation_history: Optional[List[dict]] = None,
) -> Iterator[str]:
    """
    Stream an LLM response token-by-token.
    Yields string chunks suitable for st.write_stream().
    """
    client = get_groq_client(api_key)
    if not client:
        yield "⚠️ Error: Groq API key is not configured. Please add it in Settings or the sidebar."
        return

    history_msgs = _build_message_history(conversation_history)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ──────────────────────────────────────────
# RAG Q&A (Non-Streaming)
# ──────────────────────────────────────────
def ask_llm(
    question: str,
    retrieved_chunks: List[dict],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """Answer a question using RAG context from retrieved chunks."""
    context = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)
    user_message = f"""Context from uploaded documents:

{context}

---

Question: {question}"""
    return _call_llm(
        RAG_SYSTEM_PROMPT, user_message, model, temperature,
        api_key=api_key, conversation_history=conversation_history,
    )


# ──────────────────────────────────────────
# RAG Q&A (Streaming)
# ──────────────────────────────────────────
def ask_llm_stream(
    question: str,
    context: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    conversation_history: Optional[List[dict]] = None,
    use_web_context: bool = False,
) -> Iterator[str]:
    """
    Stream a RAG answer token-by-token.

    Args:
        question            : User's query.
        context             : Combined text context (from docs or web).
        use_web_context     : If True, uses the web-search system prompt.
    """
    system = WEB_RAG_SYSTEM_PROMPT if use_web_context else RAG_SYSTEM_PROMPT
    user_message = f"""Context:

{context}

---

Question: {question}"""
    yield from _call_llm_stream(
        system, user_message, model, temperature,
        api_key=api_key, conversation_history=conversation_history,
    )


# ──────────────────────────────────────────
# SQL Generation
# ──────────────────────────────────────────
def generate_sql(
    question: str,
    columns: List[str],
    sample_data: str = "",
    model: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None,
) -> str:
    """Generate a SQL query for the given question and table schema."""
    user_message = f"""Table name: data_table
Columns: {', '.join(columns)}
Sample data:
{sample_data}

Question: {question}"""
    return _call_llm(SQL_SYSTEM_PROMPT, user_message, model, api_key=api_key)


# ──────────────────────────────────────────
# Pandas Code Generation
# ──────────────────────────────────────────
def generate_pandas_code(
    question: str,
    columns: List[str],
    dtypes: str = "",
    sample_data: str = "",
    model: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None,
) -> str:
    """Generate pandas code to answer a data question."""
    user_message = f"""DataFrame columns: {', '.join(columns)}
Data types:
{dtypes}
Sample data (first 5 rows):
{sample_data}

Question: {question}"""
    return _call_llm(PANDAS_SYSTEM_PROMPT, user_message, model, api_key=api_key)


# ──────────────────────────────────────────
# Business Insights
# ──────────────────────────────────────────
def generate_insights(
    data_summary: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> str:
    """Generate AI-powered business insights from a data summary."""
    user_message = f"""Analyze this dataset and provide business insights:

{data_summary}"""
    return _call_llm(INSIGHTS_SYSTEM_PROMPT, user_message, model, temperature, api_key=api_key)


# ──────────────────────────────────────────
# Code Extraction Helpers
# ──────────────────────────────────────────
def extract_sql(response: str) -> Optional[str]:
    """Extract SQL code from an LLM response."""
    match = re.search(r"```sql\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for SELECT ... ;
    match = re.search(r"(SELECT\s+.+?;)", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from an LLM response."""
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


CHART_SUGGESTION_SYSTEM_PROMPT = """You are an expert Data Visualisation and Business Intelligence Analyst.
Your task is to analyze the columns, data types, and sample data of a dataset, and suggest the 4 most valuable, meaningful, and business-insightful charts to display.

Rules:
1. Return ONLY a valid JSON array of objects. Do not include markdown code block backticks (like ```json), explanations, or extra text. Just raw JSON.
2. Each object in the array MUST have the exact keys:
   - "type": the chart type. Allowed values: "bar", "line", "pie", "histogram", "scatter", "area", "box", "heatmap", "treemap", "sunburst"
   - "params": a dictionary containing the Plotly express parameters.
     - For "bar", "line", "area", "scatter": {"x": "<column_name>", "y": "<column_name>", "color": "<optional_column_name>"}
     - For "pie": {"names": "<categorical_column_name>", "values": "<numeric_column_name>"}
     - For "histogram": {"x": "<column_name>"} (nbins and color are optional)
     - For "box": {"x": "<optional_column_name>", "y": "<column_name>"}
     - For "heatmap": {} (it automatically calculates numeric correlation)
     - For "treemap", "sunburst": {"path": ["<col1>", "<col2>"], "values": "<numeric_col>"}
   - "title": A professional business-oriented title for the chart.
3. Ensure every column name matches EXACTLY with the columns provided. Do not invent column names.
4. Pick combinations of columns that actually make business sense (e.g., Sales over time, Category distribution, Correlation of price/quantity, Top performers).
5. Never return empty keys or mismatched parameters.
"""


def suggest_charts_with_ai(
    columns: List[str],
    dtypes: str,
    sample_data: str,
    model: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None,
) -> List[dict]:
    """Get AI-driven chart suggestions for the dataset."""
    user_message = f"""DataFrame columns: {', '.join(columns)}
Data types:
{dtypes}
Sample data (first 5 rows):
{sample_data}

Please generate the 4 most business-insightful charts for this dataset."""

    response = _call_llm(CHART_SUGGESTION_SYSTEM_PROMPT, user_message, model, temperature=0.1, api_key=api_key)

    # Strip any backticks or surrounding markers
    cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.MULTILINE).strip()

    try:
        suggestions = json.loads(cleaned_json)
        if isinstance(suggestions, list):
            validated = []
            for s in suggestions:
                if isinstance(s, dict) and "type" in s and "params" in s and "title" in s:
                    validated.append(s)
            return validated
    except Exception:
        pass

    return []