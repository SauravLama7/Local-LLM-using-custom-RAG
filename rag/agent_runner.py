from rag.agent import decide_tool
from rag.rag_chain import get_prompt
from rag.llm import generate

def run_agent(query: str, model: str, history: list, filter_source = None ):
    """
    Main agent loop.
    Returns: (response_generator, sources, context_chunks, tools_used)
    """
    
    # Decision for which tool to use
    decision = decide_tool(query, model)
    tool = decision.get("tool","rag_search")
    reason = decision.get("reason", "")
    print(f"🤖Agent chose: {tool} - {reason}")

    # Run the chosen tool
    if tool == "clarify":
        clarify_question = decision.get("question","Could you clarify your question?")
        def clarify_gen():
            yield clarify_question
        return clarify_gen(), [], [], "clarify"
    
    elif tool == "direct_answer":
        prompt = f"""
You are a helpful assistant. Answer this question directly:

Question:
{query}

Answer:
""".strip()
        return generate(prompt, model = model), [], [], "direct_answer"
    
    #Default search
    else: 
        prompt, sources, context_chunks = get_prompt(

            query = query,
            model = model,
            history = history,
            filter_source = filter_source
        )
        
        return generate(prompt, model = model), sources , context_chunks, "rag_search"