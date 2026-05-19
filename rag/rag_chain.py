from rag.retriever import retrieve_docs

SYSTEM_PROMPT = """
You are a helpful AI assistant for answering based on the provided context.
- Respond to greetings like "hello" and "hi"
- Use information given in the context to give answers.
- If the answer is not in the context atleast try unless it is irrelevant then say "I'dont know"
- Do NOT guess, assume, or use outside knowledge.
- Be concise, clear, and factual.
Your goal is to help users understand documents accurately and reliably.
"""

def build_prompt(query, context):
    return f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""

def get_prompt(query):
    docs = retrieve_docs(query)
    context = "\n\n".join(docs)
    return build_prompt(query,context)