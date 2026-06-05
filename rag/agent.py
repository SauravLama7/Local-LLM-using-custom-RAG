import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

AGENT_SYSTEM_PROMPT = """
You are an intelligent agent that decides how to answer user questions.

You have access to these tools:
1. rag_search   — search the document knowledge base for relevant information
2. direct_answer — answer directly from your knowledge without searching
3. clarify       — ask the user for clarification when the query is too vague

Given the user query, respond ONLY with a JSON object like this:
{
  "tool": "rag_search",
  "reason": "The question is about company documents"
}

or:
{
  "tool": "direct_answer",
  "reason": "This is a general knowledge question not related to documents"
}

or:
{
  "tool": "clarify",
  "reason": "The query is too vague to search effectively",
  "question": "Could you clarify what you mean by X?"
}

Rules:
- Use rag_search for anything related to company data, employees, budgets, policies
- Use direct_answer for greetings, general knowledge, math, coding questions
- Use clarify when the query is too vague or ambiguous
- Respond ONLY with valid JSON, no extra text
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