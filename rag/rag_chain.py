from rag.retriever import retrieve_docs
from rag.prompts import SYSTEM_PROMPTS

DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant for answering based ONLY on the provided context.

Rules:
- Use ONLY the given context.
- After each fact or sentence cite the source using [1], [2] etc matching the context numbers.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Do not guess or use outside knowledge.
- Be concise, clear, and factual.
"""

def build_prompt(query, context, system_prompt, history = None):

    chat_history = ""

    if history:
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            chat_history += f"{role.upper()}:{content}\n"

    return f"""
{system_prompt}

Chat History:
{chat_history}

Context:
{context}

Question:
{query}

Answer:
""".strip()

def get_prompt(query, model = "default", history = None):
    results = retrieve_docs(query)

    # Build numbered context with source labels
    context_parts = []
    sources = []
    context_chunks = []

    for i,r in enumerate(results, start = 1):
        text = r["text"]
        source = r["metadata"].get("source", "unknown")

        context_parts.append(f"[{i}] {text}")
        context_chunks.append(text)

        # Duplicate sources
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)
    system_prompt = SYSTEM_PROMPTS.get(model,DEFAULT_SYSTEM_PROMPT)
    prompt = build_prompt(query, context, system_prompt, history)

    # Context window uses
    word_count = len(prompt.split())
    token_count = int(word_count * 1.3) # Rough estimaition
    ctx_limit = 2048
    usage_pct = round((token_count/ctx_limit) * 100, 1)
    print(f"📊 Prompt words:  {word_count}")
    print(f"📊 Approx tokens: {token_count}")
    print(f"📊 Context limit: {ctx_limit}")
    print(f"📊 Usage: {usage_pct}%")


    return prompt, sources, context_chunks