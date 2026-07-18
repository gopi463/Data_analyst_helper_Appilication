"""
auth.py — Authentication Module
AI Data Analyst Assistant

Handles user registration, login, password hashing, and session management.
"""

import bcrypt
import streamlit as st
import re
from typing import Optional
import database as db


# ──────────────────────────────────────────
# Password Utilities
# ──────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


# ──────────────────────────────────────────
# Validation
# ──────────────────────────────────────────
def validate_email(email: str) -> bool:
    """Check if email format is valid."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength. Returns (is_valid, error_message)."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit."
    return True, ""


# ──────────────────────────────────────────
# Auth Logic
# ──────────────────────────────────────────
def register_user(username: str, email: str, password: str, full_name: str = "") -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success, message).
    """
    username = username.strip()
    email = email.strip().lower()
    full_name = full_name.strip()

    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, and underscores."
    if not validate_email(email):
        return False, "Invalid email address."
    is_valid, pw_msg = validate_password(password)
    if not is_valid:
        return False, pw_msg

    if db.get_user_by_username(username):
        return False, "Username already exists."
    if db.get_user_by_email(email):
        return False, "Email already registered."

    hashed = hash_password(password)
    success = db.create_user(username, email, hashed, full_name)
    if success:
        return True, "Account created successfully!"
    return False, "Registration failed. Please try again."


def login_user(username: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """
    Authenticate a user.
    Returns (success, message, user_dict).
    """
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required.", None

    user = db.get_user_by_username(username)
    if not user:
        return False, "Invalid username or password.", None

    if not verify_password(password, user["password"]):
        return False, "Invalid username or password.", None

    db.update_last_login(user["id"])
    return True, f"Welcome back, {user.get('full_name') or username}!", user


# ──────────────────────────────────────────
# Session State Helpers
# ──────────────────────────────────────────
def is_logged_in() -> bool:
    """Check if a user is currently authenticated in session state."""
    return st.session_state.get("authenticated", False) and st.session_state.get("user") is not None


def set_logged_in(user: dict) -> None:
    """Persist user authentication to Streamlit session state."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = user
    st.session_state["user_id"] = user["id"]
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "home"


def logout() -> None:
    """Clear authentication from session state."""
    keys_to_clear = [
        "authenticated", "user", "user_id", "current_page",
        "chat_messages", "session_id", "uploaded_data",
        "vector_store", "chunks", "current_df", "current_sql_db"
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


# ──────────────────────────────────────────
# Streamlit Auth UI
# ──────────────────────────────────────────
def render_auth_page() -> None:
    """
    Render the login/register page.
    This is the gating page shown before authentication.
    """
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <div style="font-size:4rem;">🤖</div>
        <h1 style="background: linear-gradient(135deg, #667eea, #764ba2);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;">
            AI Data Analyst Assistant
        </h1>
        <p style="color:#8892a4; font-size:1.1rem; margin-bottom:2rem;">
            Enterprise-grade AI-powered data analysis platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login", "✨ Register"])

    with tab1:
        _render_login_form()

    with tab2:
        _render_register_form()


def _render_login_form() -> None:
    """Render the login form."""
    with st.form("login_form", clear_on_submit=False):
        st.markdown("### Sign In to Your Account")
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")

        if submitted:
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                with st.spinner("Authenticating..."):
                    success, message, user = login_user(username, password)
                if success:
                    set_logged_in(user)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


def _render_register_form() -> None:
    """Render the registration form."""
    with st.form("register_form", clear_on_submit=True):
        st.markdown("### Create a New Account")
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("👤 Full Name", placeholder="John Doe")
        with col2:
            username = st.text_input("🆔 Username", placeholder="johndoe")
        email = st.text_input("📧 Email", placeholder="john@example.com")
        col3, col4 = st.columns(2)
        with col3:
            password = st.text_input("🔒 Password", type="password", placeholder="Min 8 chars, 1 uppercase, 1 digit")
        with col4:
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Repeat password")
        submitted = st.form_submit_button("✨ Create Account", use_container_width=True, type="primary")

        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Creating account..."):
                    success, message = register_user(username, email, password, full_name)
                if success:
                    st.success(f"✅ {message} Please login.")
                else:
                    st.error(message)
