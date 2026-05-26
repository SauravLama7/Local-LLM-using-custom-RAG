from rag.embedding import embed
import numpy as np

def check_hallucination(answer: str, context_chunks: list[str], threshold: float = 0.3) -> dict:
    """
    Compare answer embedding against context chunks.
    Return confidence score and flag.
    """
    if not answer.strip() or not context_chunks:
        return{"score": 0.0, "grounded": False}
    
    answer_vec = embed(answer)[0]
    context_vecs = embed(context_chunks)

    # Consine similarity between answer and each chunk
    similarities = np.dot(context_vecs, answer_vec)
    max_score = float(np.max(similarities))

    return {
        "score": round(max_score, 2),
        "grounded": max_score >= threshold
    }
