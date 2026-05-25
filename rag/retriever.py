from rag.embedding import embed
from rag.vectordb import get_collection
from rag.reranker import rerank
from rag.bm25_store import bm25_search
from rag.fusion import reciprocal_rank_fusion


def retrieve_docs(query, k=10, rerank_k=5):
    collection = get_collection()

    # Embed query
    query_vec = embed(query)[0]

    # Vector search
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include = ["documents", "metadatas"]
    )

    vector_docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    # BM25 Search
    bm25_docs = bm25_search(query , top_k = k)

    # Fuse result (RRF)
    fused_docs = reciprocal_rank_fusion(vector_docs, bm25_docs) 

    # doc -> metadata lookup
    meta_lookup = {doc: meta for doc, meta in zip(vector_docs, metadatas)}

    # Rerank documents
    reranked_docs = rerank(
        query = query,
        documents = fused_docs,
        top_k = rerank_k
    )

    return[
        {"text": doc, "metadata": meta_lookup.get(doc, {"source": "unknown", "chunk_id": 0})}
        for doc in reranked_docs
    ]