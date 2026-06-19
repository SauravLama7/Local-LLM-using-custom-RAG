from rag.embedding import embed
from rag.vectordb import get_collection
from rag.reranker import rerank
from rag.bm25_store import bm25_search
from rag.fusion import reciprocal_rank_fusion
from concurrent.futures import ThreadPoolExecutor


def retrieve_docs(query, k=10, rerank_k=5, filter_source=None, allowed_sources=None):
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
        # Support both single string and list
        if isinstance(filter_source, list):
            query_params["where"] = {"source": {"$in": filter_source}} 
        else:
            query_params["where"] = {"source": filter_source}
    elif allowed_sources is not None:
        query_params["where"] = {"source": {"$in": allowed_sources}}

    # Running vector + BM25 in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(
            collection.query,
            **query_params
        )
        bm25_future = executor.submit(bm25_search, query, k)

        results               = vector_future.result()
        bm25_docs, bm25_metas = bm25_future.result()

    vector_docs = results["documents"][0]
    metadatas   = results["metadatas"][0]
    meta_lookup = {doc: meta for doc, meta in zip(vector_docs, metadatas)}

    # BM25 metadata to meta_lookup for unknown source
    for doc, meta in zip(bm25_docs, bm25_metas):
        if doc not in meta_lookup:
            meta_lookup[doc] = meta

    # Filter BM25 results — only one block needed 
    if filter_source:
        selected = filter_source if isinstance(filter_source, list) else [filter_source]
        bm25_docs = [
            doc for doc in bm25_docs
            if doc in meta_lookup and
            meta_lookup[doc].get("source") in selected
        ]
        print(f"🔎 BM25 filtered to {len(bm25_docs)} chunks from {selected}")
    elif allowed_sources is not None:
        bm25_docs = [
            doc for doc in bm25_docs
            if doc in meta_lookup and
            meta_lookup[doc].get("source") in allowed_sources
        ]
        print(f"🔎 BM25 filtered to {len(bm25_docs)} chunks from allowed sources")

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