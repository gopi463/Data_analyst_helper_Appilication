"""
history.py — Chat History Manager
AI Data Analyst Assistant
"""

import uuid
import streamlit as st
import database as db


def get_or_create_session_id() -> str:
    """Retrieve or generate a new session ID for the current chat session."""
    if st.session_state.get("session_id") is None:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def new_session() -> str:
    """Start a fresh chat session."""
    session_id = str(uuid.uuid4())
    st.session_state["session_id"] = session_id
    st.session_state["chat_messages"] = []
    return session_id


def save_message(user_id: int, role: str, content: str, sources: list = None) -> None:
    """Save a message to both session state and the database."""
    session_id = get_or_create_session_id()
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    st.session_state["chat_messages"].append({
        "role": role,
        "content": content,
        "sources": sources or [],
    })
    db.save_message(user_id, session_id, role, content, sources)


def load_session(user_id: int, session_id: str) -> list[dict]:
    """Load a previous session's messages into current session state."""
    st.session_state["session_id"] = session_id
    messages = db.get_chat_history(user_id, session_id)
    st.session_state["chat_messages"] = [
        {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
        for m in messages
    ]
    return st.session_state["chat_messages"]


def get_all_sessions(user_id: int) -> list[dict]:
    """Get all unique sessions for the user."""
    return db.get_user_sessions(user_id)


def delete_session(user_id: int, session_id: str) -> None:
    """Delete a session from the database and clear from state if active."""
    db.delete_session(user_id, session_id)
    if st.session_state.get("session_id") == session_id:
        new_session()


def clear_all_history(user_id: int) -> None:
    """Delete all chat history for the user."""
    db.delete_all_history(user_id)
    new_session()


def search_history(user_id: int, query: str) -> list[dict]:
    """Search the user's chat history by keyword."""
    return db.search_history(user_id, query)


def get_current_messages() -> list[dict]:
    """Return the current session's messages from session state."""
    return st.session_state.get("chat_messages", [])
