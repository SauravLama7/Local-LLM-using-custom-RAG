import streamlit as st
from auth.db import verify_user, init_db

def show_login_page():
    """Display login page and handle authentication."""

    # Initialize DB on first run
    init_db()

    st.title("🧠 Local LLM RAG Assistant")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 Sign In")

        tab1, tab2 = st.tabs(["Login", "Continue as Guest"])

        # Login tab
        with tab1:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            if st.button("Login", use_container_width=True):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user      = user
                        st.session_state.is_guest  = False
                        st.success(f"Welcome, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

        # Guest tab
        with tab2:
            st.info("Continue without logging in. Chat won't be saved.")
            if st.button("Continue as Guest", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user      = {"username": "guest", "role": "guest", "name": "Guest"}
                st.session_state.is_guest  = True
                st.rerun()