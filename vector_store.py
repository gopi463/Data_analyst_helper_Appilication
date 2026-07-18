"""
vector_store.py — FAISS Vector Store
AI Data Analyst Assistant

Per-user FAISS index with metadata persistence.
"""

import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Optional
from config import DATA_DIR


# ──────────────────────────────────────────
# VectorStore Class
# ──────────────────────────────────────────
class VectorStore:
    """
    FAISS-based vector store with chunk metadata.

    Each user gets an isolated index, stored in:
        data/<user_id>/faiss.index
        data/<user_id>/metadata.pkl
    """

    def __init__(self, dimension: int, user_id: Optional[int] = None):
        self.dimension = dimension
        self.user_id = user_id
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine on normalized vecs)
        self.metadata: List[dict] = []

    # ─── CRUD ─────────────────────────────
    def add_vectors(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        """Add embeddings and their corresponding metadata chunks."""
        embeddings = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        """
        Return top-k most similar chunks for a query embedding.

        Args:
            query_embedding : shape (1, dim) normalized float32 array
            k               : number of results

        Returns:
            List of chunk dicts with added 'score' field.
        """
        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        query_embedding = np.asarray(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                chunk = dict(self.metadata[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results

    def reset(self) -> None:
        """Clear all vectors and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    # ─── Persistence ──────────────────────
    def _get_paths(self) -> tuple[Path, Path]:
        """Return index and metadata file paths scoped to the user."""
        base = DATA_DIR / str(self.user_id) if self.user_id else DATA_DIR
        base.mkdir(parents=True, exist_ok=True)
        return base / "faiss.index", base / "metadata.pkl"

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path, meta_path = self._get_paths()
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"metadata": self.metadata, "dimension": self.dimension}, f)

    def load(self) -> bool:
        """Load a saved index and metadata. Returns True if successful."""
        index_path, meta_path = self._get_paths()
        if index_path.exists() and meta_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(meta_path, "rb") as f:
                    data = pickle.load(f)
                self.metadata = data.get("metadata", [])
                self.dimension = data.get("dimension", self.dimension)
                return True
            except Exception:
                return False
        return False

    def exists(self) -> bool:
        """Check if a saved index exists for this user."""
        index_path, meta_path = self._get_paths()
        return index_path.exists() and meta_path.exists()


# ──────────────────────────────────────────
# Factory / Session Helpers
# ──────────────────────────────────────────
def build_vector_store(chunks: List[dict], embeddings: np.ndarray, user_id: int = None) -> VectorStore:
    """Build a VectorStore from chunks and their embeddings."""
    dim = embeddings.shape[1]
    store = VectorStore(dimension=dim, user_id=user_id)
    store.add_vectors(embeddings, chunks)
    return store