"""
retriever.py — Hybrid RAG Retriever with Reranking
AI Data Analyst Assistant

Features:
- Hybrid Search: BM25 keyword search + FAISS semantic search
- Reciprocal Rank Fusion (RRF) for score combination
- Score-fusion reranking for final ordering
"""

import math
from typing import List, Tuple, Optional

from embedder import embed_query
from vector_store import VectorStore
from config import DEFAULT_TOP_K


# ──────────────────────────────────────────
# BM25 Retriever
# ──────────────────────────────────────────
class BM25Retriever:
    """
    Lightweight BM25 keyword retriever over a list of chunk dicts.

    Uses the `rank-bm25` library. If unavailable, falls back to
    a simple TF-based scorer so the app never crashes.
    """

    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        self._bm25 = None
        self._tokenized = None
        if not chunks:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._tokenized = [
                c["text"].lower().split() for c in chunks
            ]
            self._bm25 = BM25Okapi(self._tokenized)
        except ImportError:
            pass  # Graceful fallback below

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> List[dict]:
        """Return top-k chunks with a 'bm25_score' field."""
        if not self.chunks:
            return []

        query_tokens = query.lower().split()

        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
        else:
            # Fallback: simple token overlap count
            scores = [
                sum(1 for t in query_tokens if t in c["text"].lower())
                for c in self.chunks
            ]

        # Pair with metadata and sort
        scored = sorted(
            zip(scores, self.chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        results = []
        for score, chunk in scored[:k]:
            c = dict(chunk)
            c["bm25_score"] = float(score)
            results.append(c)
        return results


# ──────────────────────────────────────────
# Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────
def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion scoring formula."""
    return 1.0 / (k + rank + 1)


def _reciprocal_rank_fusion(
    faiss_results: List[dict],
    bm25_results: List[dict],
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    Args:
        faiss_results : Chunks ranked by FAISS cosine similarity.
        bm25_results  : Chunks ranked by BM25 score.
        faiss_weight  : Weight for semantic (FAISS) scores.
        bm25_weight   : Weight for keyword (BM25) scores.

    Returns:
        Merged list of unique chunks sorted by fused RRF score (descending).
    """
    # Use (file, chunk_index) as a unique chunk key
    chunk_scores: dict = {}
    chunk_map: dict = {}

    for rank, chunk in enumerate(faiss_results):
        key = (chunk.get("file", ""), chunk.get("chunk_index", rank))
        chunk_scores[key] = chunk_scores.get(key, 0.0) + faiss_weight * _rrf_score(rank)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(bm25_results):
        key = (chunk.get("file", ""), chunk.get("chunk_index", rank))
        chunk_scores[key] = chunk_scores.get(key, 0.0) + bm25_weight * _rrf_score(rank)
        if key in chunk_map:
            # Merge: Keep existing FAISS fields (like 'score') and add bm25_score
            existing = dict(chunk_map[key])
            existing["bm25_score"] = chunk.get("bm25_score", 0.0)
            chunk_map[key] = existing
        else:
            chunk_map[key] = chunk

    # Sort by combined score descending
    sorted_keys = sorted(chunk_scores, key=lambda k: chunk_scores[k], reverse=True)
    fused = []
    for key in sorted_keys:
        c = dict(chunk_map[key])
        c["rrf_score"] = round(chunk_scores[key], 4)
        fused.append(c)

    return fused


# ──────────────────────────────────────────
# Reranker (Score Fusion)
# ──────────────────────────────────────────
def _normalize(values: List[float]) -> List[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rerank_results(
    chunks: List[dict],
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """
    Re-score and re-order chunks using normalized FAISS score + BM25 score.

    Expects chunks to have 'score' (FAISS cosine) and optionally 'bm25_score'.
    """
    if not chunks:
        return []

    faiss_scores = [c.get("score", 0.0) for c in chunks]
    bm25_scores  = [c.get("bm25_score", 0.0) for c in chunks]

    norm_faiss = _normalize(faiss_scores)
    norm_bm25  = _normalize(bm25_scores)

    reranked = []
    for i, chunk in enumerate(chunks):
        combined = 0.65 * norm_faiss[i] + 0.35 * norm_bm25[i]
        c = dict(chunk)
        c["rerank_score"] = round(combined, 4)
        reranked.append(c)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ──────────────────────────────────────────
# Hybrid Retrieve (Main Entry Point)
# ──────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    vector_store: VectorStore,
    k: int = DEFAULT_TOP_K,
    bm25_retriever: Optional[BM25Retriever] = None,
) -> List[dict]:
    """
    Perform hybrid retrieval: FAISS semantic + BM25 keyword → RRF fusion → rerank.

    Args:
        query           : User's natural-language query.
        vector_store    : Loaded FAISS VectorStore instance.
        k               : Number of final results to return.
        bm25_retriever  : Pre-built BM25Retriever. Built on-the-fly if None.

    Returns:
        Top-k reranked chunk dicts.
    """
    # ── FAISS semantic search ──
    query_embedding = embed_query(query)
    faiss_results = vector_store.search(query_embedding, k=k * 2)

    # ── BM25 keyword search ──
    if bm25_retriever is None:
        all_chunks = vector_store.get_all_chunks()
        bm25_retriever = BM25Retriever(all_chunks)

    bm25_results = bm25_retriever.search(query, k=k * 2)

    # ── Fuse results ──
    fused = _reciprocal_rank_fusion(faiss_results, bm25_results)

    # ── Rerank & return top-k ──
    return rerank_results(fused, top_k=k)


# ──────────────────────────────────────────
# Legacy pure-vector retrieve (kept for backwards compat)
# ──────────────────────────────────────────
def retrieve(
    query: str,
    vector_store: VectorStore,
    k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """Pure FAISS retrieval (no BM25). Kept for backwards compatibility."""
    query_embedding = embed_query(query)
    return vector_store.search(query_embedding, k=k)


# ──────────────────────────────────────────
# Context Builder
# ──────────────────────────────────────────
MIN_CONFIDENCE_SCORE = 0.25   # Chunks below this threshold are considered low-quality

def retrieve_with_context(
    query: str,
    vector_store: VectorStore,
    k: int = DEFAULT_TOP_K,
    bm25_retriever: Optional[BM25Retriever] = None,
) -> Tuple[str, List[dict], bool]:
    """
    Retrieve chunks via hybrid search and format them into a combined context string.

    Returns:
        (context_string, source_list, is_high_confidence)
        is_high_confidence=False signals the caller to try a web search fallback.
    """
    chunks = hybrid_retrieve(query, vector_store, k=k, bm25_retriever=bm25_retriever)
    if not chunks:
        return "", [], False

    # Determine confidence: use the raw FAISS score (absolute cosine similarity)
    best_score = chunks[0].get("score", 0.0)
    is_high_confidence = best_score >= MIN_CONFIDENCE_SCORE

    context_parts = []
    sources = []
    seen_sources = set()

    for chunk in chunks:
        context_parts.append(chunk["text"])
        source_key = (chunk.get("file", ""), chunk.get("page", 0))
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "file":  chunk.get("file", "Unknown"),
                "page":  chunk.get("page", 0),
                "type":  chunk.get("type", "unknown"),
                "score": round(chunk.get("rerank_score", chunk.get("score", 0.0)), 3),
            })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources, is_high_confidence


# ──────────────────────────────────────────
# Source Formatter
# ──────────────────────────────────────────
def format_sources(sources: List[dict]) -> str:
    """Format source list into a readable citation string."""
    if not sources:
        return ""
    lines = ["\n📎 **Sources:**"]
    for s in sources:
        page_info = f" (Page {s['page']})" if s.get("page") else ""
        score_info = f" · score: {s['score']}" if s.get("score") else ""
        lines.append(f"- 📄 `{s['file']}`{page_info}{score_info}")
    return "\n".join(lines)