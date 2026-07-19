"""
tests/test_database.py — Unit Tests for database.py
AI Data Analyst Assistant

Uses a temporary in-memory/temp-file SQLite DB to avoid touching production data.
"""
import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Patch DB_PATH before importing database ───────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def temp_db(tmp_path_factory):
    """Redirect the database to a temp file for testing."""
    db_file = tmp_path_factory.mktemp("db") / "test_app.db"
    with patch("database.DB_PATH", db_file):
        import database as db
        db.init_db()
        yield db


# ── User Tests ────────────────────────────────────────────────────────────────
class TestUserOperations:
    def test_create_user_success(self, temp_db):
        result = temp_db.create_user("testuser", "test@example.com", "hashed_pw", "Test User")
        assert result is True

    def test_create_duplicate_user_fails(self, temp_db):
        temp_db.create_user("dupeuser", "dupe@example.com", "hashed_pw")
        result = temp_db.create_user("dupeuser", "dupe@example.com", "hashed_pw")
        assert result is False

    def test_get_user_by_username(self, temp_db):
        temp_db.create_user("fetchuser", "fetch@example.com", "pw")
        user = temp_db.get_user_by_username("fetchuser")
        assert user is not None
        assert user["username"] == "fetchuser"

    def test_get_nonexistent_user_returns_none(self, temp_db):
        user = temp_db.get_user_by_username("nobody_at_all")
        assert user is None

    def test_get_user_by_email(self, temp_db):
        temp_db.create_user("emailuser", "email@example.com", "pw")
        user = temp_db.get_user_by_email("email@example.com")
        assert user is not None
        assert user["email"] == "email@example.com"


# ── Chat History Tests ────────────────────────────────────────────────────────
class TestChatHistory:
    def setup_method(self):
        """Create a fresh user for each test."""
        import database as db
        import uuid
        self.db = db
        self.username = f"chatuser_{uuid.uuid4().hex[:6]}"
        db.create_user(self.username, f"{self.username}@x.com", "pw")
        user = db.get_user_by_username(self.username)
        self.user_id = user["id"]
        self.session_id = str(uuid.uuid4())

    def test_save_and_retrieve_message(self):
        self.db.save_message(self.user_id, self.session_id, "user", "Hello!")
        history = self.db.get_chat_history(self.user_id, self.session_id)
        assert len(history) == 1
        assert history[0]["content"] == "Hello!"
        assert history[0]["role"] == "user"

    def test_sources_saved_as_list(self):
        sources = [{"file": "report.pdf", "page": 1}]
        self.db.save_message(self.user_id, self.session_id, "assistant", "Answer.", sources)
        history = self.db.get_chat_history(self.user_id, self.session_id)
        assert isinstance(history[0]["sources"], list)
        assert history[0]["sources"][0]["file"] == "report.pdf"

    def test_multiple_messages_ordered_asc(self):
        self.db.save_message(self.user_id, self.session_id, "user", "First")
        self.db.save_message(self.user_id, self.session_id, "assistant", "Second")
        history = self.db.get_chat_history(self.user_id, self.session_id)
        assert history[0]["content"] == "First"
        assert history[1]["content"] == "Second"

    def test_search_history_finds_match(self):
        self.db.save_message(self.user_id, self.session_id, "user", "unique_keyword_xyz")
        results = self.db.search_history(self.user_id, "unique_keyword_xyz")
        assert len(results) >= 1

    def test_delete_session(self):
        self.db.save_message(self.user_id, self.session_id, "user", "to be deleted")
        self.db.delete_session(self.user_id, self.session_id)
        history = self.db.get_chat_history(self.user_id, self.session_id)
        assert len(history) == 0


# ── Feedback Tests ────────────────────────────────────────────────────────────
class TestFeedback:
    def setup_method(self):
        import database as db
        import uuid
        self.db = db
        username = f"fbuser_{uuid.uuid4().hex[:6]}"
        db.create_user(username, f"{username}@x.com", "pw")
        user = db.get_user_by_username(username)
        self.user_id = user["id"]
        self.session_id = str(uuid.uuid4())

    def test_save_thumbs_up(self):
        self.db.save_feedback(self.user_id, self.session_id, 0, "up")
        stats = self.db.get_feedback_stats(self.user_id)
        assert stats["thumbs_up"] >= 1

    def test_save_thumbs_down(self):
        self.db.save_feedback(self.user_id, self.session_id, 1, "down")
        stats = self.db.get_feedback_stats(self.user_id)
        assert stats["thumbs_down"] >= 1

    def test_feedback_stats_structure(self):
        stats = self.db.get_feedback_stats(self.user_id)
        assert "thumbs_up" in stats
        assert "thumbs_down" in stats


# ── Settings Tests ────────────────────────────────────────────────────────────
class TestUserSettings:
    def setup_method(self):
        import database as db
        import uuid
        self.db = db
        username = f"settuser_{uuid.uuid4().hex[:6]}"
        db.create_user(username, f"{username}@x.com", "pw")
        user = db.get_user_by_username(username)
        self.user_id = user["id"]

    def test_get_default_settings(self):
        settings = self.db.get_user_settings(self.user_id)
        assert "model" in settings
        assert "temperature" in settings

    def test_update_settings(self):
        self.db.update_user_settings(self.user_id, temperature=0.5)
        settings = self.db.get_user_settings(self.user_id)
        assert abs(settings["temperature"] - 0.5) < 0.01

    def test_update_multiple_settings(self):
        self.db.update_user_settings(self.user_id, top_k=8, chunk_size=800)
        settings = self.db.get_user_settings(self.user_id)
        assert settings["top_k"] == 8
        assert settings["chunk_size"] == 800
