"""
web_search.py — Web Search Fallback
AI Data Analyst Assistant

Used when RAG retrieval returns no results or low-confidence context.
Queries DuckDuckGo (no API key required) and returns structured snippets.
"""

from typing import List


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Perform a DuckDuckGo text search and return structured results.

    Args:
        query       : The search query string.
        max_results : Maximum number of results to return.

    Returns:
        List of dicts: [{"title": ..., "url": ..., "snippet": ...}]
        Returns an empty list on any failure so callers can degrade gracefully.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except ImportError:
        # Library not installed — silently skip
        return []
    except Exception:
        # Network error, rate-limit, etc. — silently skip
        return []


def format_web_results_as_context(results: List[dict]) -> str:
    """
    Convert web search results into a plain-text context block
    suitable for passing to the LLM as supplementary context.
    """
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Web Result {i}] {r['title']}\n"
            f"Source: {r['url']}\n"
            f"{r['snippet']}"
        )
    return "\n\n---\n\n".join(parts)


def format_web_sources(results: List[dict]) -> str:
    """
    Format web results as a citation block for display in the chat UI.
    """
    if not results:
        return ""
    lines = ["\n🌐 **Answered from Web Search:**"]
    for r in results:
        lines.append(f"- [{r['title']}]({r['url']})")
    return "\n".join(lines)
