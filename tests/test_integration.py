"""
tests/test_integration.py — Integration Tests for AI Data Analyst Assistant
AI Data Analyst Assistant

Verifies the end-to-end integration between:
1. Ingestion -> Vector Store -> Hybrid Retrieval
2. Retrieve -> Web Fallback trigger
3. Database -> Chat Session -> Message feedback
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Stub sentence_transformers and embedder to make integration test fast and offline-friendly
_fake_st = MagicMock()
_fake_embedder = MagicMock()
# Mock embed_query and embed_chunks to return mock vectors
_fake_embedder.embed_chunks.return_value = np.random.rand(3, 384).astype("float32")
_fake_embedder.embed_query.return_value = np.random.rand(1, 384).astype("float32")

sys.modules.setdefault("sentence_transformers", _fake_st)
sys.modules.setdefault("embedder", _fake_embedder)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chunker
import vector_store
import retriever
import web_search
import database as db


# ── Integration Fixtures ──────────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def temp_integration_db(tmp_path_factory):
    """Redirect the database to a temporary location for integration testing."""
    temp_dir = tmp_path_factory.mktemp("integration")
    db_file = temp_dir / "integration_app.db"
    
    # Patch both DB_PATH in database and DATA_DIR in vector_store to isolate files
    with patch("database.DB_PATH", db_file), \
         patch("vector_store.DATA_DIR", temp_dir), \
         patch("config.DATA_DIR", temp_dir):
        db.init_db()
        yield db


# ── Integration Tests ─────────────────────────────────────────────────────────
def test_end_to_end_ingestion_and_hybrid_retrieval(tmp_path_factory):
    """
    Test 1: Documents -> Chunks -> FAISS & BM25 Index -> Hybrid Search.
    """
    # 1. Prepare raw documents simulating a PDF or CSV upload
    raw_docs = [
        {"text": "Python is a high-level programming language used for data science and web development.", "file": "intro.pdf", "page": 1, "type": "pdf"},
        {"text": "FAISS is a library for efficient similarity search of dense vectors.", "file": "faiss_guide.pdf", "page": 2, "type": "pdf"},
        {"text": "Streamlit allows you to build data apps in minutes using pure Python.", "file": "streamlit.pdf", "page": 1, "type": "pdf"},
    ]

    # 2. Chunk documents
    chunks = chunker.chunk_documents(raw_docs, chunk_size=300, overlap=50)
    assert len(chunks) == 3

    # 3. Create dummy embeddings matching the stub dimension (384)
    dummy_embeddings = np.random.rand(len(chunks), 384).astype("float32")

    # 4. Build Vector Store
    user_id = 999
    vs = vector_store.build_vector_store(chunks, dummy_embeddings, user_id=user_id)
    assert vs.total_vectors == 3

    # 5. Build BM25 Index
    bm25 = retriever.BM25Retriever(chunks)

    # 6. Perform a Hybrid Query (expecting high confidence)
    # Patch the embedding query generation so it matches our search space
    with patch("retriever.embed_query", return_value=dummy_embeddings[2:3]):
        context, sources, is_confident = retriever.retrieve_with_context(
            query="Streamlit data apps",
            vector_store=vs,
            k=2,
            bm25_retriever=bm25
        )
        
        assert "Streamlit" in context
        assert len(sources) >= 1
        assert sources[0]["file"] == "streamlit.pdf"
        assert is_confident is True


def test_hybrid_search_to_web_fallback_integration():
    """
    Test 2: Retrieve with low confidence -> Trigger Web Fallback search.
    """
    # 1. Create a Vector Store with unrelated documents
    raw_docs = [{"text": "Apples are delicious red fruits.", "file": "fruits.pdf", "page": 1, "type": "pdf"}]
    chunks = chunker.chunk_documents(raw_docs, chunk_size=100, overlap=10)
    dummy_embeddings = np.random.rand(1, 384).astype("float32")
    vs = vector_store.build_vector_store(chunks, dummy_embeddings, user_id=888)
    bm25 = retriever.BM25Retriever(chunks)

    # 2. Ask a question completely unrelated to apples (e.g. quantum computing)
    # The FAISS similarity score will be extremely low or negative, triggering low confidence
    with patch("retriever.embed_query", return_value=np.zeros((1, 384), dtype="float32")):
        context, sources, is_confident = retriever.retrieve_with_context(
            query="What is quantum mechanics?",
            vector_store=vs,
            k=1,
            bm25_retriever=bm25
        )
        
        # Verify retriever signals low confidence
        assert is_confident is False

    # 3. Simulate UI workflow in app.py when is_confident is False
    mock_web_results = [
        {"title": "Quantum Mechanics Intro", "url": "https://physics.org/qm", "snippet": "Quantum mechanics describes particles at the atomic scale."}
    ]
    
    with patch("web_search.web_search", return_value=mock_web_results):
        web_results = web_search.web_search("What is quantum mechanics?", max_results=1)
        assert len(web_results) == 1
        assert web_results[0]["title"] == "Quantum Mechanics Intro"
        
        # Build context from web results
        web_context = web_search.format_web_results_as_context(web_results)
        assert "physics.org/qm" in web_context


def test_database_feedback_and_chat_session_integration(temp_integration_db):
    """
    Test 3: Create User -> Start Session -> Send Message -> Record Thumbs Up/Down.
    """
    # 1. Create user
    username = "integration_test_user"
    email = "integration@test.com"
    temp_integration_db.create_user(username, email, "password123", "Integration Tester")
    user = temp_integration_db.get_user_by_username(username)
    assert user is not None
    user_id = user["id"]

    # 2. Start a chat session
    session_id = "test_session_xyz"
    
    # 3. Save User message
    temp_integration_db.save_message(user_id, session_id, "user", "What is the sales forecast?")
    
    # 4. Save Assistant Response
    temp_integration_db.save_message(user_id, session_id, "assistant", "The sales forecast is up 12% next quarter.")
    
    # 5. Verify chat history retrieved
    history = temp_integration_db.get_chat_history(user_id, session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    # 6. Save message feedback on assistant's response (message index 1)
    temp_integration_db.save_feedback(user_id, session_id, message_idx=1, rating="up")

    # 7. Get feedback stats and verify
    stats = temp_integration_db.get_feedback_stats(user_id)
    assert stats["thumbs_up"] == 1
    assert stats["thumbs_down"] == 0
