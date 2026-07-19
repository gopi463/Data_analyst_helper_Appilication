"""
database.py — SQLite Database Manager
AI Data Analyst Assistant

Tables:
  - users           : User accounts
  - chat_history    : Per-user conversation logs
  - uploaded_files  : Metadata of uploaded files per user
  - reports         : Generated report metadata
  - user_settings   : Per-user configuration
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from config import DB_PATH


# ──────────────────────────────────────────
# Connection Helper
# ──────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ──────────────────────────────────────────
# Schema Initialization
# ──────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()

    # Users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            full_name   TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT (datetime('now')),
            last_login  TEXT
        )
    """)

    # Chat History
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            session_id  TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            sources     TEXT    DEFAULT '[]',
            created_at  TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Uploaded Files
    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            filename    TEXT    NOT NULL,
            file_type   TEXT    NOT NULL,
            file_size   INTEGER DEFAULT 0,
            row_count   INTEGER DEFAULT 0,
            col_count   INTEGER DEFAULT 0,
            uploaded_at TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Reports
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            title       TEXT    NOT NULL,
            report_type TEXT    DEFAULT 'PDF',
            file_path   TEXT,
            created_at  TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # User Settings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER UNIQUE NOT NULL,
            theme           TEXT    DEFAULT 'dark',
            model           TEXT    DEFAULT 'llama-3.3-70b-versatile',
            temperature     REAL    DEFAULT 0.1,
            top_k           INTEGER DEFAULT 5,
            chunk_size      INTEGER DEFAULT 600,
            embedding_model TEXT    DEFAULT 'all-MiniLM-L6-v2',
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Message Feedback
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            session_id  TEXT    NOT NULL,
            message_idx INTEGER NOT NULL,
            rating      TEXT    NOT NULL CHECK(rating IN ('up', 'down')),
            created_at  TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Recruiter Inquiries
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recruiter_inquiries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            company    TEXT,
            email      TEXT    NOT NULL,
            message    TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


# ──────────────────────────────────────────
# User Operations
# ──────────────────────────────────────────
def create_user(username: str, email: str, hashed_password: str, full_name: str = "") -> bool:
    """Insert a new user. Returns True on success, False if duplicate."""
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO users (username, email, password, full_name) VALUES (?, ?, ?, ?)",
            (username, email, hashed_password, full_name)
        )
        conn.commit()
        # Create default settings for the new user
        user = get_user_by_username(username)
        if user:
            conn.execute(
                "INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)",
                (user["id"],)
            )
            conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_username(username: str) -> Optional[dict]:
    """Fetch user dict by username."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    """Fetch user dict by email."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_last_login(user_id: int) -> None:
    """Update the last_login timestamp for a user."""
    conn = get_connection()
    conn.execute(
        "UPDATE users SET last_login = datetime('now') WHERE id = ?", (user_id,)
    )
    conn.commit()
    conn.close()


# ──────────────────────────────────────────
# Chat History Operations
# ──────────────────────────────────────────
def save_message(user_id: int, session_id: str, role: str, content: str, sources: list = None) -> None:
    """Persist a chat message to the database."""
    sources_json = json.dumps(sources or [])
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_history (user_id, session_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
        (user_id, session_id, role, content, sources_json)
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id: int, session_id: str = None) -> list[dict]:
    """Retrieve chat history for a user, optionally filtered by session."""
    conn = get_connection()
    if session_id:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE user_id = ? AND session_id = ? ORDER BY created_at ASC",
            (user_id, session_id)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["sources"] = json.loads(d.get("sources", "[]"))
        result.append(d)
    return result


def get_user_sessions(user_id: int) -> list[dict]:
    """Return unique session IDs with first message preview for a user."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT session_id,
               MIN(created_at) as started_at,
               COUNT(*) as message_count
        FROM chat_history
        WHERE user_id = ?
        GROUP BY session_id
        ORDER BY started_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_session(user_id: int, session_id: str) -> None:
    """Delete all messages in a session for a user."""
    conn = get_connection()
    conn.execute(
        "DELETE FROM chat_history WHERE user_id = ? AND session_id = ?",
        (user_id, session_id)
    )
    conn.commit()
    conn.close()


def delete_all_history(user_id: int) -> None:
    """Delete all chat history for a user."""
    conn = get_connection()
    conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def search_history(user_id: int, query: str) -> list[dict]:
    """Full-text search across chat history for a user."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chat_history WHERE user_id = ? AND content LIKE ? ORDER BY created_at DESC LIMIT 50",
        (user_id, f"%{query}%")
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────
# File Metadata Operations
# ──────────────────────────────────────────
def save_file_metadata(user_id: int, filename: str, file_type: str,
                        file_size: int = 0, row_count: int = 0, col_count: int = 0) -> None:
    """Record an uploaded file's metadata."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO uploaded_files (user_id, filename, file_type, file_size, row_count, col_count) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, filename, file_type, file_size, row_count, col_count)
    )
    conn.commit()
    conn.close()


def get_user_files(user_id: int) -> list[dict]:
    """Fetch all file metadata for a user."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM uploaded_files WHERE user_id = ? ORDER BY uploaded_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────
# Reports Operations
# ──────────────────────────────────────────
def save_report(user_id: int, title: str, file_path: str, report_type: str = "PDF") -> None:
    """Save report metadata."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO reports (user_id, title, file_path, report_type) VALUES (?, ?, ?, ?)",
        (user_id, title, file_path, report_type)
    )
    conn.commit()
    conn.close()


def get_user_reports(user_id: int) -> list[dict]:
    """Fetch all report metadata for a user."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────
# Settings Operations
# ──────────────────────────────────────────
def get_user_settings(user_id: int) -> dict:
    """Get settings for a user, creating defaults if none exist."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    if not row:
        conn.execute("INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)", (user_id,))
        conn.commit()
        row = conn.execute(
            "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    conn.close()
    return dict(row) if row else {}


def update_user_settings(user_id: int, **kwargs: Any) -> None:
    """Update specific settings fields for a user."""
    if not kwargs:
        return
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [user_id]
    conn = get_connection()
    conn.execute(f"UPDATE user_settings SET {fields} WHERE user_id = ?", values)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────
# Stats
# ──────────────────────────────────────────
def get_user_stats(user_id: int) -> dict:
    """Return aggregate statistics for a user's activity."""
    conn = get_connection()
    total_messages = conn.execute(
        "SELECT COUNT(*) FROM chat_history WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total_files = conn.execute(
        "SELECT COUNT(*) FROM uploaded_files WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total_reports = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    total_sessions = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM chat_history WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    conn.close()
    return {
        "total_messages": total_messages,
        "total_files": total_files,
        "total_reports": total_reports,
        "total_sessions": total_sessions,
    }


# ──────────────────────────────────────────
# Feedback Operations
# ──────────────────────────────────────────
def save_feedback(user_id: int, session_id: str, message_idx: int, rating: str) -> None:
    """Persist a user feedback rating ('up' or 'down') for a specific message."""
    conn = get_connection()
    # Upsert: replace any prior rating for the same message
    conn.execute(
        """
        INSERT INTO feedback (user_id, session_id, message_idx, rating)
        VALUES (?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        (user_id, session_id, message_idx, rating)
    )
    conn.commit()
    conn.close()


def get_feedback_stats(user_id: int) -> dict:
    """Return aggregate thumbs-up / thumbs-down counts for a user."""
    conn = get_connection()
    up = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE user_id = ? AND rating = 'up'", (user_id,)
    ).fetchone()[0]
    down = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE user_id = ? AND rating = 'down'", (user_id,)
    ).fetchone()[0]
    conn.close()
    return {"thumbs_up": up, "thumbs_down": down}


# ──────────────────────────────────────────
# Recruiter Inquiry Operations
# ──────────────────────────────────────────
def save_inquiry(name: str, company: str, email: str, message: str) -> None:
    """Persist a contact form message from a recruiter."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO recruiter_inquiries (name, company, email, message) VALUES (?, ?, ?, ?)",
        (name, company, email, message)
    )
    conn.commit()
    conn.close()


def get_inquiries() -> list[dict]:
    """Retrieve all recruiter inquiries order by newest first."""
    conn = get_connection()
    cur = conn.execute("SELECT * FROM recruiter_inquiries ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "company": r["company"],
            "email": r["email"],
            "message": r["message"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
