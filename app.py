import streamlit as st
import base64
from rag.rag_chain import get_prompt
from rag.llm import generate, VISION_MODELS
from memory.chat_store import load_chat, save_chat

# PAGE CONFIG
st.set_page_config(
    page_title="Local LLM using RAG",
    page_icon="🧠",
    layout="wide"
)

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("### Model Info")
    st.info("Please select the model you want to use.")

    MODEL_OPTIONS = ["llama3.2:1b", "ministral-3:3b"]
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

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
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

        try:
            prompt = get_prompt(
                query=user_input,
                model=selected_model,
                history=st.session_state.messages
            )

            with st.spinner("🔍 Searching documents + thinking..."):
                for token in generate(
                    prompt,
                    model=selected_model,
                    image_bytes=uploaded_image   
                ):
                    full_response += token
                    placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.error(full_response)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "model": selected_model
    })
    save_chat(selected_model, st.session_state.messages)