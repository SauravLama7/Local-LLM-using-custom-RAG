from rag.embedding import embed
from rag.vectordb import get_collection
from rag.reranker import rerank
from rag.bm25_store import bm25_search
from rag.fusion import reciprocal_rank_fusion
from concurrent.futures import ThreadPoolExecutor


def retrieve_docs(query, k=10, rerank_k=5, filter_source=None):
    collection = get_collection()

    # Embed query
    query_vec = embed(query)[0]

    # Only add where clause if filter is actually set — never pass where=None
    query_params = {
        "query_embeddings": [query_vec],
        "n_results": k,
        "include": ["documents", "metadatas"]
    }
    if filter_source:
        query_params["where"] = {"source": filter_source}

    # Running vector + BM25 in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(
            collection.query,
            **query_params       
        )

        bm25_future = executor.submit(bm25_search, query, k)

        results   = vector_future.result()
        bm25_docs = bm25_future.result()

    vector_docs = results["documents"][0]
    metadatas   = results["metadatas"][0]
    meta_lookup = {doc: meta for doc, meta in zip(vector_docs, metadatas)}

    # Fuse result (RRF)
    fused_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)

    # Rerank documents
    reranked_docs = rerank(
        query=query,
        documents=fused_docs,
        top_k=rerank_k
    )

    return [
        {"text": doc, "metadata": meta_lookup.get(doc, {"source": "unknown", "chunk_id": 0})}
        for doc in reranked_docs
    ]