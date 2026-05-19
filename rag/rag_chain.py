from rag.retriever import retrieve_docs
from rag.prompts import SYSTEM_PROMPTS

DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant for answering based ONLY on the provided context.

Rules:
- Use ONLY the given context.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Do not guess or use outside knowledge.
- Be concise, clear, and factual.
"""

def build_prompt(query, context, system_prompt):
    return f"""
{system_prompt}

Context:
{context}

Question:
{query}

Answer:
""".strip()

def get_prompt(query, model = "default"):
    docs = retrieve_docs(query)
    context = "\n\n".join(docs)
    system_prompt = SYSTEM_PROMPTS.get(model,DEFAULT_SYSTEM_PROMPT)
    return build_prompt(query,context,system_prompt)