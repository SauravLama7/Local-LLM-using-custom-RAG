from rag.agent import decide_tool
from rag.rag_chain import get_prompt
from rag.llm import generate
from rag.hallucination_guard import check_hallucination
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def rephrase_query(query: str, model: str) -> str:
    """Ask the LLM to rephrase the query for better retrieval."""
    payload = {
        "model": model,
        "prompt": f"""Rephrase this question in a different way to help search a document database better.
Return ONLY the rephrased question, no explanation.

Original: {query}
Rephrased:""",
        "stream": False,
        "options": {"num_ctx": 256, "temperature": 0.7}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        rephrased = response.json().get("response", "").strip()
        print(f"🔄 Rephrased query: {rephrased}")
        return rephrased if rephrased else query
    except Exception as e:
        print(e)
        return query  # fallback to original


def run_agent(query: str, model: str, history: list, filter_source=None, allowed_sources=None,
              max_retries: int = 1, stream_callback=None, image_bytes: bytes = None,
              status_callback=None):  # ← added status_callback
    """
    Main agent loop with self-correction.
    stream_callback(token): called for each token on the FIRST attempt only.
    status_callback(msg): called to update UI status messages.
    Returns: (full_response, sources, context_chunks, tool_used, attempts, was_streamed)
    """

    # If image provided, skip agent routing — answer directly with image
    if image_bytes:
        print("🖼️ Image detected — routing to direct_answer")
        if status_callback:
            status_callback("🖼️ Analyzing image...")
        image_prompt = f"""You are a helpful assistant. Analyze the provided image and answer the question.

Question: {query}

Answer:""".strip()
        full_response = ""
        for token in generate(image_prompt, model=model, image_bytes=image_bytes):
            full_response += token
            if stream_callback:
                stream_callback(token)
        # If no response generated, return a fallback
        if not full_response.strip():
            full_response = "I was unable to analyze the image. Please try again."
            if stream_callback:
                stream_callback(full_response)
        return full_response, [], [], {}, "direct_answer", 1, True

    # Step 1: Decide which tool to use
    if status_callback:
        status_callback("🧠 Deciding which tool to use...")
    decision = decide_tool(query, model)
    tool     = decision.get("tool", "rag_search")
    reason   = decision.get("reason", "")
    print(f"🤖 Agent chose: {tool} — {reason}")

    # Non-RAG tools don't need self-correction — stream directly
    if tool == "clarify":
        if status_callback:
            status_callback("❓ Asking for clarification...")
        clarify_question = decision.get("question", "Could you clarify your question?")
        if stream_callback:
            stream_callback(clarify_question)
        return clarify_question, [], [], {}, "clarify", 1, True

    elif tool == "direct_answer":
        if status_callback:
            status_callback("💡 Generating direct answer...")
        prompt = f"""You are a helpful assistant. Answer this question directly in english.

Question: {query}

Answer:""".strip()
        full_response = ""
        for token in generate(prompt, model=model):
            full_response += token
            if stream_callback:
                stream_callback(token)
        return full_response, [], [], {}, "direct_answer", 1, True

    # RAG search with self-correction loop
    current_query = query
    best_response = None
    best_sources  = []
    best_chunks   = []
    best_indexed = {}
    best_score    = -1
    attempts      = 0
    was_streamed  = False

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        print(f"🔁 Attempt {attempts}/{max_retries + 1} — query: {current_query[:60]}")

        # Get prompt + context
        if status_callback:
            status_callback("🔍 Searching documents...")
        prompt, sources, context_chunks, indexed_chunks = get_prompt(
            query=current_query,
            model=model,
            history=history,
            filter_source=filter_source,
            allowed_sources=allowed_sources
        )

        # Generate response
        full_response = ""

        if attempt == 0:
            # First attempt: stream live to the UI
            if status_callback:
                status_callback("✍️ Generating answer...")
            for token in generate(prompt, model=model):
                full_response += token
                if stream_callback:
                    stream_callback(token)
            was_streamed = True
        else:
            # Retry attempt: generate silently, no live streaming
            if status_callback:
                status_callback(f"🔄 Low confidence — retrying with rephrased query (attempt {attempts})...")
            for token in generate(prompt, model=model):
                full_response += token
            was_streamed = False

        # Check quality
        result = check_hallucination(full_response, context_chunks)
        score  = result["score"]
        print(f"   Score: {score} | Grounded: {result['grounded']}")

        # Keep best response
        if score > best_score:
            best_score    = score
            best_response = full_response
            best_sources  = sources
            best_chunks   = context_chunks
            best_indexed = indexed_chunks

        # If grounded enough, stop retrying
        if result["grounded"]:
            print(f"✅ Good answer on attempt {attempts}")
            break

        # If not last attempt, rephrase and retry
        if attempt < max_retries:
            print(f"🔄 Low score ({score}), rephrasing and retrying...")
            current_query = rephrase_query(current_query, model)

    # was_streamed only True if the BEST response came from the streamed first attempt
    return best_response, best_sources, best_chunks, best_indexed, "rag_search", attempts, was_streamed and attempts == 1