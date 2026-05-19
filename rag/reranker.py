from sentence_transformers import CrossEncoder

_model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, documents, top_k = 5):
    """
    query: str
    documents: list[str]
    returns: list[str]
    """
    if not documents:
        return[]
    
    pairs = [(query,doc) for doc in documents]

    scores = _model.predict(pairs)

    ranked = sorted(
        zip(documents,scores),
        key = lambda x: x[1],
        reverse = True
    )

    return [doc for doc, _ in ranked[:top_k]]

