import os
from pathlib import Path
import streamlit as st
import base64
from auth.login_page import show_login_page
from auth.db import get_allowed_docs, init_db, log_query
from rag.agent_runner import run_agent
from rag.llm import VISION_MODELS
from memory.chat_store import load_chat, save_chat
from rag.hallucination_guard import check_hallucination


# PAGE CONFIG
st.set_page_config(
    page_title="Local LLM using RAG",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
        section.main > div {
        padding-bottom: 10px;
        }
        .stChatFloatingInputContainer{
        position: fixed;
         bottom: 0;
        }
    </style>
""", unsafe_allow_html=True)

# Auth gate
init_db()  # ensure DB exists
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    show_login_page()
    st.stop()

# Get current user info
current_user = st.session_state.user
is_guest     = st.session_state.is_guest

# Guest query limit
GUEST_QUERY_LIMIT = 10  # max queries per session

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Settings")

    # User info
    st.markdown("### 👤 User")
    st.success(f"**{current_user['name']}**")
    st.caption(f"Role: `{current_user['role']}`")
    if st.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Guest query limit indicator
    if is_guest:
        remaining = GUEST_QUERY_LIMIT - st.session_state.get("guest_query_count", 0)
        if remaining > 3:
            st.info(f"💬 {remaining} queries remaining this session.")
        elif remaining > 0:
            st.warning(f"⚠️ Only {remaining} queries remaining this session.")
        else:
            st.error("❌ Query limit reached. Please log in for unlimited access.")

    # Admin Panel — only for admin role
    if current_user["role"] == "admin":
        st.markdown("### 🛠️ Admin Panel")
        with st.expander("👥 Manage Users"):

            # Add new user
            st.markdown("#### Add User")
            new_username = st.text_input("Username", key="new_username")
            new_password = st.text_input("Password", type="password", key="new_password")
            new_name     = st.text_input("Full Name", key="new_name")
            new_role     = st.selectbox("Role", ["admin", "hr", "dev", "finance", "guest"], key="new_role")

            if st.button("➕ Add User"):
                if not new_username or not new_password or not new_name:
                    st.error("All fields required.")
                else:
                    from auth.db import add_user
                    success = add_user(new_username, new_password, new_name, new_role)
                    if success:
                        st.success(f"✅ User '{new_username}' added.")
                    else:
                        st.error(f"❌ Username '{new_username}' already exists.")

            st.divider()

            # View and delete users
            st.markdown("#### Current Users")
            from auth.db import get_all_users, delete_user
            users = get_all_users()
            for u in users:
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.caption(f"**{u['name']}** (`{u['username']}`)")
                col2.caption(f"`{u['role']}`")
                if u["username"] != "admin":  # prevent deleting admin
                    if col3.button("🗑️", key=f"del_user_{u['username']}"):
                        delete_user(u["username"])
                        st.toast(f"Deleted {u['username']}")
                        st.rerun()

        # Audit viewer
        with st.expander("📋 Audit Log"):
            from auth.db import get_audit_log

            col1, col2 = st.columns([1, 1])
            with col1:
                log_limit = st.slider("Show last N entires", 10, 500, 50, step = 10)
            with col2:
                all_users = ["All Users"] + [u["username"] for u in get_all_users()]
                filter_user = st.selectbox("Filter by user", all_users, key = "audit_user_filter")

            logs = get_audit_log(limit=log_limit)

            if filter_user != "All Users":
                logs = [l for l in logs if l["username"] == filter_user]
                
            if logs:
                for entry in logs:
                    grounded_icon = "✅" if entry["grounded"] else "⚠️"
                    st.caption(
                        f"{entry['timestamp']} | "
                        f"**{entry['username']}** (`{entry['role']}`) | "
                        f"{entry['tool_used']} | "
                        f"{grounded_icon} {entry['score']} | "
                        f"{entry['query'][:60]}"
                    )
                    if entry["sources"]:
                        st.caption(f"  📄 {entry['sources']}")
                    st.divider()
            else:
                st.caption("No queries logged yet.")

    st.markdown("### Model Info")
    st.info("qwen2.5:3b and ministral-3:3b are the two models available. ministral-3:3b also supports vision")

    MODEL_OPTIONS = ["qwen2.5:3b", "ministral-3:3b"]
    selected_model = st.sidebar.selectbox("Choose Model", MODEL_OPTIONS)

    # Switch chats when switching model
    if "current_model" not in st.session_state:
        st.session_state.current_model = selected_model
    if st.session_state.current_model != selected_model:
        st.session_state.current_model = selected_model
        st.session_state.messages = load_chat(current_user["username"], selected_model)

    st.sidebar.success(f"🧠 Active Model: {selected_model}")

    # Image uploader (vision models only)
    uploaded_image = None
    if selected_model in VISION_MODELS:
        st.markdown("### 🖼️ Image Input")
        uploaded_file = st.file_uploader(
            "Upload an image to analyze",
            type=["png", "jpg", "jpeg", "webp"],
            help="Attach an image for the model to analyze with your next message."
        )
        if uploaded_file:
            uploaded_image = uploaded_file.read()
            st.image(uploaded_image, caption="Image ready to send", use_container_width=True)

    # Add Document Feature — hidden for guests
    if not is_guest:
        st.markdown("### 📄 Add New Documents")
        doc_files = st.file_uploader(
            "Upload documents to knowledge base",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            key="doc_uploader"
        )

        if doc_files:
            if st.button("⚡ Ingest Documents"):
                import shutil
                from scripts.ingest import ingest

                os.makedirs("data/raw", exist_ok=True)
                saved   = []
                skipped = []

                for doc in doc_files:
                    dest = os.path.join("data/raw", doc.name)

                    # Don't overwrite if identical file already exists
                    if os.path.exists(dest):
                        skipped.append(doc.name)
                        continue

                    with open(dest, "wb") as f:
                        shutil.copyfileobj(doc, f)
                    saved.append(doc.name)

                if saved:
                    with st.spinner(f"⚙️ Ingesting {len(saved)} file(s)..."):
                        try:
                            ingest()
                            st.success(f"✅ Ingested: {', '.join(saved)}")
                        except Exception as e:
                            st.error(f"❌ Ingest failed: {str(e)}")

                if skipped:
                    st.info(f"⏭️ Already exists: {', '.join(skipped)}")

    st.markdown("### 🗂️ Knowledge Base")
    raw_path = Path("data/raw")
    if raw_path.exists():
        files = list(raw_path.glob("*"))
        if files:
            for f in files:
                col1, col2 = st.columns([3, 1])
                col1.caption(f"📄 {f.name}")
                if not is_guest:
                    if col2.button("🗑️", key=f"del_{f.name}"):
                        # Delete from disk
                        f.unlink()
                        # Delete from chromaDB
                        from rag.vectordb import delete_by_source
                        delete_by_source(f.name)
                        # Rebuild BM25
                        from rag.bm25_store import rebuild_bm25_without
                        rebuild_bm25_without(f.name)
                        # Remove from hash store so re-adding works
                        from rag.hash_store import load_hashes, save_hashes
                        hashes = load_hashes()
                        # Remove by filepath
                        updated = {k: v for k, v in hashes.items() if f.name not in k}
                        save_hashes(updated)
                        st.toast(f"Deleted {f.name}")
                        st.rerun()
        else:
            st.caption("No documents yet")

    # Filter source — filtered by role
    st.markdown("🔎 Filter Source")
    raw_path        = Path("data/raw")
    all_files       = [f.name for f in raw_path.glob("*")] if raw_path.exists() else []
    available_files = get_allowed_docs(current_user["role"], all_files)  # ← role filtered
    selected_filter = st.multiselect(
        "Search within document",
        available_files,
        placeholder="Leave empty to search all documents"
    )
    filter_source = selected_filter if selected_filter else None

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.pop("sources", None)
        if not is_guest:
            save_chat(current_user["username"], selected_model, [])
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "Local LLM using RAG, ChromaDB, all-MiniLM-L6-v2 embeddings, "
        "and Ollama backend. Supports vision input on ministral-3:3b."
    )

# TITLE
st.title("🧠 Local LLM RAG Assistant")
st.caption("Ask questions over your documents using local AI")

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = [] if is_guest else load_chat(current_user["username"], selected_model)

# Guest query count init 
if is_guest and "guest_query_count" not in st.session_state:
    st.session_state.guest_query_count = 0

# Pre-warm model
if "model_warmed" not in st.session_state or st.session_state.current_model != st.session_state.get("warmed_model"):
    import threading
    def warm():
        try:
            import requests
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": selected_model, "prompt": " ", "keep_alive": "30m"},
                timeout=30
            )
        except:
            pass
    threading.Thread(target=warm, daemon=True).start()
    st.session_state.model_warmed = True
    st.session_state.warmed_model = selected_model


# Helper: render a single message bubble
def render_message(msg):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.caption(f"🧠 {msg.get('model', '')}")

        # If the message carried an image, show it above the text
        if "image_b64" in msg:
            img_bytes = base64.b64decode(msg["image_b64"])
            st.image(img_bytes, width=300)

        st.markdown(msg["content"])

        # Hallucination indicator
        if "grounded" in msg:
            if msg["grounded"]:
                st.caption(f"✅ Grounded ({msg['confidence']})")
            else:
                st.warning(f"⚠️ Low confidence ({msg['confidence']}) — answer may not be grounded in your documents.")

        # Show citations if present
        if msg.get("sources"):
            citation_md = " ".join([f"`📄 {src}`" for src in msg["sources"]])
            st.markdown(f"**Sources:** {citation_md}")


# DISPLAY CHAT HISTORY
for msg in st.session_state.messages:
    render_message(msg)


# USER INPUT
user_input = st.chat_input("Ask something about your documents...")

# MAIN LOGIC
if user_input:

    # Check guest query limit 
    if is_guest:
        if st.session_state.guest_query_count >= GUEST_QUERY_LIMIT:
            st.warning("❌ You've reached the guest query limit. Please log in for unlimited access.")
            st.stop()
        # ── Limit message length for guests
        if len(user_input) > 500:
            st.warning("⚠️ Message too long. Guest queries are limited to 500 characters.")
            st.stop()
        st.session_state.guest_query_count += 1

    # Build user message — attach image if one was uploaded
    user_msg = {"role": "user", "content": user_input}
    if uploaded_image:
        user_msg["image_b64"] = base64.b64encode(uploaded_image).decode("utf-8")

    st.session_state.messages.append(user_msg)

    # Don't save chat for guests
    if not is_guest:
        save_chat(current_user["username"], selected_model, st.session_state.messages)

    render_message(user_msg)

    # Generate response
    with st.chat_message("assistant"):
        st.caption(f"🧠 Using model: {selected_model}")
        placeholder    = st.empty()
        full_response  = ""
        sources        = []
        context_chunks = []
        tool_used      = "rag_search"
        attempts       = 1
        result         = {"grounded": False, "score": 0.0}

        try:
            placeholder_text   = []
            status_placeholder = st.empty()  # ← live status display

            def on_token(token):
                placeholder_text.append(token)
                placeholder.markdown("".join(placeholder_text))

            def on_status(msg):
                status_placeholder.caption(f"⏳ {msg}")

            full_response, sources, context_chunks, tool_used, attempts, was_streamed = run_agent(
                query=user_input,
                model=selected_model,
                history=st.session_state.messages,
                filter_source=filter_source,
                allowed_sources=available_files if current_user["role"] != "admin" else None,
                stream_callback=on_token,
                image_bytes=uploaded_image,
                status_callback=on_status
            )

            # Clear status once done
            status_placeholder.empty()

            # If retry happened, the streamed text doesn't match final answer — overwrite
            if not was_streamed:
                placeholder.markdown(full_response)

            tool_labels = {
                "rag_search":    "🔍 Searched documents",
                "direct_answer": "💡 Answered directly",
                "clarify":       "❓ Asking clarification"
            }
            label = tool_labels.get(tool_used, "")
            if attempts > 1:
                label += f" (retried {attempts - 1}x)"
            st.caption(label)

            # Hallucination check — only for rag_search
            if tool_used == "rag_search" and context_chunks:
                result = check_hallucination(full_response, context_chunks)
                if not result["grounded"]:
                    st.warning(f"⚠️ Low confidence ({result['score']}) — answer may not be grounded in your documents.")
                else:
                    st.caption(f"✅ Grounded ({result['score']})")

            # Log query to audit log
            if not is_guest:
                log_query(
                    username=current_user["username"],
                    role=current_user["role"],
                    query=user_input,
                    tool_used=tool_used,
                    grounded=result["grounded"],
                    score=result["score"],
                    sources=sources,
                    model=selected_model
                )

            # Render Citations
            if sources:
                citation_md = " ".join([f"`📄 {src}`" for src in sources])
                st.markdown(f"**Sources:** {citation_md}")

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.error(full_response)

    # Save assistant message
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    full_response,
        "model":      selected_model,
        "sources":    sources,
        "grounded":   result["grounded"],
        "confidence": result["score"],
        "tool_used":  tool_used,
        "attempts":   attempts
    })

    # Don't save chat for guests
    if not is_guest:
        save_chat(current_user["username"], selected_model, st.session_state.messages)