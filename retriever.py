"""
retriever.py — RAG Retriever with Citation Support
AI Data Analyst Assistant
"""

from typing import List, Tuple
from embedder import embed_query
from vector_store import VectorStore
from config import DEFAULT_TOP_K


def retrieve(
    query: str,
    vector_store: VectorStore,
    k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """
    Retrieve top-k most relevant chunks for a query.

    Returns:
        List of chunk dicts including 'score', 'file', 'page', 'text'.
    """
    query_embedding = embed_query(query)
    results = vector_store.search(query_embedding, k=k)
    return results


def retrieve_with_context(
    query: str,
    vector_store: VectorStore,
    k: int = DEFAULT_TOP_K,
) -> Tuple[str, List[dict]]:
    """
    Retrieve chunks and format them into a combined context string.

    Returns:
        (context_string, source_list)
    """
    chunks = retrieve(query, vector_store, k=k)
    if not chunks:
        return "", []

    context_parts = []
    sources = []
    seen_sources = set()

    for chunk in chunks:
        context_parts.append(chunk["text"])
        source_key = (chunk.get("file", ""), chunk.get("page", 0))
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "file": chunk.get("file", "Unknown"),
                "page": chunk.get("page", 0),
                "type": chunk.get("type", "unknown"),
                "score": round(chunk.get("score", 0.0), 3),
            })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def format_sources(sources: List[dict]) -> str:
    """Format source list into a readable citation string."""
    if not sources:
        return ""
    lines = ["\n📎 **Sources:**"]
    for s in sources:
        page_info = f" (Page {s['page']})" if s.get("page") else ""
        lines.append(f"- 📄 `{s['file']}`{page_info}")
    return "\n".join(lines)