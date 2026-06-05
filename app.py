import os
from pathlib import Path
import streamlit as st
import base64
from rag.agent_runner import run_agent
from rag.rag_chain import get_prompt
from rag.llm import generate, VISION_MODELS
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
""", unsafe_allow_html = True)

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("### Model Info")
    st.info("qwen2.5:3b and ministral-3:3b are the two models available. ministral-3:3b also supports vision")

    MODEL_OPTIONS = ["qwen2.5:3b", "ministral-3:3b"]
    selected_model = st.sidebar.selectbox("Choose Model", MODEL_OPTIONS)

    # Switch chats when switching model
    if "current_model" not in st.session_state:
        st.session_state.current_model = selected_model
    if st.session_state.current_model != selected_model:
        st.session_state.current_model = selected_model
        st.session_state.messages = load_chat(selected_model)

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

    # Add Document Feature
    st.markdown("### 📄 Add New Documents")
    doc_files = st.file_uploader(
        """Upload documents to knowledge base""",
        type = ["pdf","txt"],
        accept_multiple_files = True,
        key = "doc_uploader"
    )

    if doc_files:
        if st.button("⚡ Ingest Documents"):
            import shutil
            from scripts.ingest import ingest

            os.makedirs("data/raw", exist_ok = True)
            saved = []
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
                col1, col2 = st.columns([3,1])
                col1.caption(f"📄 {f.name}")
                if col2.button("🗑️", key=f"del_{f.name}"):
                    #Delete from disk 
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
    
    st.markdown("🔎 Filter Source")
    raw_path = Path("data/raw")
    available_files = [f.name for f in raw_path.glob("*")] if raw_path.exists() else []
    filter_source = st.selectbox(
        "Search within document",
        ["All Documents"] + available_files
    )
    filter_source = None if filter_source == "All Documents" else filter_source
                

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.pop("sources", None)
        save_chat(selected_model, [])
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
    st.session_state.messages = load_chat(selected_model)

# Pre-warm model
if "model_warmed" not in st.session_state or st.session_state.current_model != st.session_state.get("warmed_model"):
    try:
        import requests
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": selected_model, "prompt": " ", "keep_alive": "30m"},
            timeout=30
        )
    except:
        pass
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

    # Build user message — attach image if one was uploaded
    user_msg = {"role": "user", "content": user_input}
    if uploaded_image:
        user_msg["image_b64"] = base64.b64encode(uploaded_image).decode("utf-8")

    st.session_state.messages.append(user_msg)
    save_chat(selected_model, st.session_state.messages)
    render_message(user_msg)

    # Generate response
    with st.chat_message("assistant"):
        st.caption(f"🧠 Using model: {selected_model}")
        placeholder = st.empty()
        full_response = ""
        sources = []
        context_chunks = []
        tool_used = "rag_search"
        result = {"grounded": False, "score": 0.0}

        try:
            with st.spinner("🤖 Agent thinking..."):
                response_gen, sources, context_chunks, tool_used = run_agent(
                    query = user_input,
                    model = selected_model,
                    history = st.session_state.messages,
                    filter_source = filter_source
                )
            
            tool_labels = {
                "rag_search": "🔍 Searched documents",
                "direct_answer": "💡 Answered directly",
                "clarify": "❓ Asking clarification"
            }
            st.caption(tool_labels.get(tool_used,""))
            for token in response_gen:
                full_response += token
                placeholder.markdown(full_response)

            # Hallucination check
            if tool_used == "rag_search" and context_chunks:
                result = check_hallucination(full_response, context_chunks)
                if not result["grounded"]:
                    st.warning(f"⚠️ Low confidence ({result['score']}) — answer may not be grounded in your documents.")
                else:
                    st.caption(f"✅ Grounded ({result['score']})")

            
            # Render Citations
            if sources:
                citation_md = " ".join([
                    f"`📄 {src}`" for src in sources                   
                ])
                st.markdown(f"**Sources:** {citation_md}")

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.error(full_response)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "model": selected_model,
        "sources": sources,
        "grounded": result["grounded"],
        "confidence": result["score"],
        "tool_used":tool_used
    })
    save_chat(selected_model, st.session_state.messages)