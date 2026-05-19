import streamlit as st
from rag.rag_chain import get_prompt
from rag.llm import generate


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
    MODEL_OPTIONS = [
        "llama3.2:1b",
        "ministral-3:3b",
    ]

    selected_model = st.sidebar.selectbox(
        "choose Model",
        MODEL_OPTIONS
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.write("Local RAG system using embeddings + vector DB + Ollama")


# TITLE
st.title("🧠 Local LLM RAG Assistant")
st.caption("Ask questions over your documents using local AI")


# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY CHAT HISTORY

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# USER INPUT
user_input = st.chat_input("Ask something about your documents...")


# MAIN LOGIC
if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            prompt = get_prompt(user_input)

            with st.spinner("🔍 Searching documents + thinking..."):
                for token in generate(prompt, model = selected_model):
                    full_response += token
                    placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            placeholder.error(full_response)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })