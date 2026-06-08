import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

AGENT_SYSTEM_PROMPT = """
You are an agent that routes questions to the right tool.

Tools:
1. rag_search    — search company documents (DEFAULT choice)
2. direct_answer — only for greetings or basic math
3. clarify       — only when query has no words to search with

IMPORTANT RULES:
- If the query mentions ANYTHING that could be in a document, use rag_search
- Questions about "the company", "employees", "budget", "policy", "department", "name", "CEO" → ALWAYS rag_search
- Greetings like "hi", "hello", "how are you" → direct_answer
- Math like "2+2" → direct_answer
- Vague single words with no context like "it", "that" → clarify
- When in doubt → rag_search

Examples:
"what is the company name?" → rag_search
"who is the CEO?" → rag_search  
"is bipin in development?" → rag_search
"hello" → direct_answer
"2+2" → direct_answer
"tell me about it" → clarify

Respond ONLY with valid JSON:
{"tool": "rag_search", "reason": "question about company"}
"""
def decide_tool(query:str, model:str) -> dict:
    """Ask the LLM which tool to use for theis query."""
    payload = {
        "model": model,
        "prompt": f"{AGENT_SYSTEM_PROMPT}\n\nUser query: {query}\n\nJSON response:",
        "stream": False,
        "options": {"num_ctx": 512, "temperature": 0}
    }

    try:
        response = requests.post(OLLAMA_URL,json = payload , timeout = 60)
        raw = response.json().get("response", "").strip()

        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0 :
            return json.loads(raw[start:end])
    except Exception as e:
        print(f"⚠️ Agent decision failed: {e}")

    # Default to RAG search if decisions fail
    return{"tool":"rag_search", "reason": "fallback"}