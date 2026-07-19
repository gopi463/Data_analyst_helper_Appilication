"""
tests/test_chunker.py — Unit Tests for chunker.py
AI Data Analyst Assistant
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunker import chunk_documents, _split_text


# ── Fixtures ────────────────────────────────────────────────────────────────
SAMPLE_DOC = {
    "text": "This is a sentence. " * 50,  # ~1000 chars
    "file": "test.csv",
    "page": 1,
    "type": "csv",
}

SHORT_DOC = {
    "text": "Short text.",
    "file": "short.pdf",
    "page": 1,
    "type": "pdf",
}

EMPTY_DOC = {
    "text": "",
    "file": "empty.pdf",
    "page": 0,
    "type": "pdf",
}


# ── Tests ────────────────────────────────────────────────────────────────────
class TestSplitText:
    def test_short_text_returns_single_chunk(self):
        result = _split_text("Hello world.", chunk_size=600, overlap=100)
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_long_text_splits_into_multiple_chunks(self):
        text = "a" * 2000
        result = _split_text(text, chunk_size=600, overlap=100)
        assert len(result) > 1

    def test_all_chunks_non_empty(self):
        text = "word " * 300
        result = _split_text(text, chunk_size=300, overlap=50)
        assert all(len(c) > 0 for c in result)

    def test_chunk_size_respected(self):
        text = "a" * 2000
        result = _split_text(text, chunk_size=200, overlap=50)
        for chunk in result:
            assert len(chunk) <= 250  # a bit of slack for sentence boundary logic


class TestChunkDocuments:
    def test_empty_list_returns_empty(self):
        result = chunk_documents([])
        assert result == []

    def test_empty_text_doc_is_skipped(self):
        result = chunk_documents([EMPTY_DOC])
        assert result == []

    def test_short_doc_produces_one_chunk(self):
        result = chunk_documents([SHORT_DOC], chunk_size=600, overlap=100)
        assert len(result) == 1
        assert result[0]["text"] == "Short text."

    def test_chunk_metadata_is_preserved(self):
        result = chunk_documents([SHORT_DOC])
        chunk = result[0]
        assert chunk["file"] == "short.pdf"
        assert chunk["page"] == 1
        assert chunk["type"] == "pdf"
        assert chunk["chunk_index"] == 0

    def test_long_doc_produces_multiple_chunks(self):
        result = chunk_documents([SAMPLE_DOC], chunk_size=200, overlap=50)
        assert len(result) > 1

    def test_chunk_index_increments(self):
        result = chunk_documents([SAMPLE_DOC], chunk_size=200, overlap=50)
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_multiple_docs(self):
        result = chunk_documents([SHORT_DOC, SHORT_DOC])
        assert len(result) == 2

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_documents([SAMPLE_DOC], chunk_size=100, overlap=100)
