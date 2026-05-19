from rag.embedding import embed
from rag.vectordb import get_collection

def retrieve_docs(query, k=4):
    collection = get_collection()

    query_vec = embed(query)[0]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k
    )
    
    return results["documents"][0]