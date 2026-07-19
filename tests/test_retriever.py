"""
tests/test_retriever.py — Unit Tests for retriever.py
AI Data Analyst Assistant
"""
import pytest
import sys
import os
from unittest.mock import MagicMock

# ── Stub out heavy ML dependencies before importing retriever ─────────────────
# retriever.py imports embedder.py which imports sentence_transformers.
# We don't need actual embeddings for BM25/RRF/reranker unit tests.
_fake_st = MagicMock()
_fake_embedder = MagicMock()
sys.modules.setdefault("sentence_transformers", _fake_st)
sys.modules.setdefault("embedder", _fake_embedder)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import (
    BM25Retriever,
    _rrf_score,
    _reciprocal_rank_fusion,
    rerank_results,
    format_sources,
)



# ── Fixtures ─────────────────────────────────────────────────────────────────
CHUNKS = [
    {"text": "Python is a programming language.", "file": "doc1.pdf", "page": 1, "type": "pdf", "chunk_index": 0},
    {"text": "Machine learning uses algorithms.", "file": "doc1.pdf", "page": 2, "type": "pdf", "chunk_index": 1},
    {"text": "Data science involves statistics.", "file": "doc2.csv", "page": 0, "type": "csv", "chunk_index": 0},
    {"text": "Neural networks are deep learning models.", "file": "doc2.csv", "page": 0, "type": "csv", "chunk_index": 1},
    {"text": "Python is used in data analysis.", "file": "doc3.pdf", "page": 1, "type": "pdf", "chunk_index": 0},
]


# ── BM25Retriever Tests ───────────────────────────────────────────────────────
class TestBM25Retriever:
    def test_init_with_chunks(self):
        bm25 = BM25Retriever(CHUNKS)
        assert bm25.chunks == CHUNKS

    def test_init_with_empty_chunks(self):
        bm25 = BM25Retriever([])
        assert bm25.chunks == []

    def test_search_returns_list(self):
        bm25 = BM25Retriever(CHUNKS)
        results = bm25.search("Python programming")
        assert isinstance(results, list)

    def test_search_empty_index_returns_empty(self):
        bm25 = BM25Retriever([])
        results = bm25.search("anything")
        assert results == []

    def test_search_respects_k_limit(self):
        bm25 = BM25Retriever(CHUNKS)
        results = bm25.search("Python", k=2)
        assert len(results) <= 2

    def test_results_have_bm25_score(self):
        bm25 = BM25Retriever(CHUNKS)
        results = bm25.search("Python")
        for r in results:
            assert "bm25_score" in r

    def test_relevant_result_ranked_higher(self):
        bm25 = BM25Retriever(CHUNKS)
        results = bm25.search("Python")
        # "Python" appears in chunks 0 and 4 — they should rank higher
        top_texts = [r["text"] for r in results[:2]]
        assert any("Python" in t for t in top_texts)


# ── RRF Score Tests ───────────────────────────────────────────────────────────
class TestRRFScore:
    def test_rank_0_score_is_highest(self):
        assert _rrf_score(0) > _rrf_score(1) > _rrf_score(10)

    def test_score_is_positive(self):
        assert _rrf_score(5) > 0

    def test_k_parameter_affects_score(self):
        assert _rrf_score(0, k=10) > _rrf_score(0, k=100)


# ── RRF Fusion Tests ──────────────────────────────────────────────────────────
class TestReciprocalRankFusion:
    def test_fused_contains_all_unique_chunks(self):
        faiss = [
            {"text": "A", "file": "f1.pdf", "chunk_index": 0, "score": 0.9},
            {"text": "B", "file": "f1.pdf", "chunk_index": 1, "score": 0.7},
        ]
        bm25 = [
            {"text": "B", "file": "f1.pdf", "chunk_index": 1, "bm25_score": 5.0},
            {"text": "C", "file": "f2.pdf", "chunk_index": 0, "bm25_score": 3.0},
        ]
        result = _reciprocal_rank_fusion(faiss, bm25)
        texts = [r["text"] for r in result]
        assert "A" in texts
        assert "B" in texts
        assert "C" in texts

    def test_fused_results_have_rrf_score(self):
        faiss = [{"text": "X", "file": "f.pdf", "chunk_index": 0, "score": 0.5}]
        bm25  = [{"text": "X", "file": "f.pdf", "chunk_index": 0, "bm25_score": 2.0}]
        result = _reciprocal_rank_fusion(faiss, bm25)
        assert all("rrf_score" in r for r in result)

    def test_empty_inputs_return_empty(self):
        assert _reciprocal_rank_fusion([], []) == []


# ── Reranker Tests ────────────────────────────────────────────────────────────
class TestRerankResults:
    def test_returns_top_k(self):
        chunks = [
            {"text": f"chunk {i}", "file": "f.pdf", "chunk_index": i,
             "score": float(i) / 10, "bm25_score": float(i)}
            for i in range(10)
        ]
        result = rerank_results(chunks, top_k=3)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        assert rerank_results([]) == []

    def test_results_have_rerank_score(self):
        chunks = [{"text": "a", "file": "f.pdf", "chunk_index": 0, "score": 0.5, "bm25_score": 1.0}]
        result = rerank_results(chunks, top_k=1)
        assert "rerank_score" in result[0]

    def test_sorted_by_rerank_score_descending(self):
        chunks = [
            {"text": "low",  "file": "f.pdf", "chunk_index": 0, "score": 0.1, "bm25_score": 0.1},
            {"text": "high", "file": "f.pdf", "chunk_index": 1, "score": 0.9, "bm25_score": 9.0},
        ]
        result = rerank_results(chunks, top_k=2)
        assert result[0]["text"] == "high"


# ── Format Sources Tests ──────────────────────────────────────────────────────
class TestFormatSources:
    def test_empty_sources_returns_empty_string(self):
        assert format_sources([]) == ""

    def test_single_source_formats_correctly(self):
        sources = [{"file": "report.pdf", "page": 3, "type": "pdf", "score": 0.87}]
        result = format_sources(sources)
        assert "report.pdf" in result
        assert "Page 3" in result

    def test_no_page_info_for_page_zero(self):
        sources = [{"file": "data.csv", "page": 0, "type": "csv", "score": 0.5}]
        result = format_sources(sources)
        assert "Page" not in result
