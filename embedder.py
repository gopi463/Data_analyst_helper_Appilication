"""
embedder.py — Embedding Engine
AI Data Analyst Assistant

Uses SentenceTransformer to generate vector embeddings for chunks and queries.
Cached model loading for performance.
"""

import numpy as np
import streamlit as st
from typing import List
from sentence_transformers import SentenceTransformer
from config import DEFAULT_EMBEDDING_MODEL


# ──────────────────────────────────────────
# Cached Model Loader
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer model.
    Cached at the resource level so it persists across reruns.
    """
    return SentenceTransformer(model_name)


# ──────────────────────────────────────────
# Chunk Embedding
# ──────────────────────────────────────────
def embed_chunks(
    chunks: List[dict],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of chunk dicts.

    Returns:
        numpy.ndarray of shape (n_chunks, embedding_dim)
    """
    model = load_embedding_model(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


# ──────────────────────────────────────────
# Query Embedding
# ──────────────────────────────────────────
def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        numpy.ndarray of shape (1, embedding_dim)
    """
    model = load_embedding_model(model_name)
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype("float32")


# ──────────────────────────────────────────
# Alias for backwards compatibility
# ──────────────────────────────────────────
def create_embeddings(chunks: List[dict], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """Alias for embed_chunks."""
    return embed_chunks(chunks, model_name)