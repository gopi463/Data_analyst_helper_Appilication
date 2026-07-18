"""
chunker.py — Smart Text Chunker
AI Data Analyst Assistant

Splits documents into overlapping text chunks while preserving metadata.
"""

from typing import List


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[dict]:
    """
    Split a list of document dicts into overlapping text chunks.

    Input:
        [{"text": "...", "file": "data.csv", "page": 1, "type": "csv"}]

    Output:
        [{"text": "...", "file": "data.csv", "page": 1, "type": "csv", "chunk_index": 0}]
    """
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap}).")

    chunks = []
    for document in documents:
        text = document.get("text", "").strip()
        if not text:
            continue
        doc_chunks = _split_text(text, chunk_size, overlap)
        for idx, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "text": chunk_text,
                "file": document.get("file", "unknown"),
                "page": document.get("page", 0),
                "type": document.get("type", "unknown"),
                "chunk_index": idx,
            })
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a single text string into overlapping chunks.
    Tries to split at sentence boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to avoid cutting mid-sentence
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.6:
                chunk = chunk[: last_period + 1]

        chunks.append(chunk.strip())
        start += max(step, 1)

    return [c for c in chunks if c]


def chunk_text(
    documents: List[dict],
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[dict]:
    """
    Alias for chunk_documents — maintains backwards compatibility
    with the original RAG system.
    """
    return chunk_documents(documents, chunk_size, overlap)