from rag.embedding import embed
from rag.vectordb import get_collection
from rag.reranker import rerank


def retrieve_docs(query, k=7, rerank_k=3):
    collection = get_collection()

    # Embed query
    query_vec = embed(query)[0]

    # Vector search
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k
    )

    docs = results["documents"][0] 

    # Rerank documents
    reranked_docs = rerank(
        query=query,
        documents=docs,
        top_k=rerank_k
    )

    return reranked_docs