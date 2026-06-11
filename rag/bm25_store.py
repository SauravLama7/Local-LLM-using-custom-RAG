import pickle
import os
from rank_bm25 import BM25Okapi

BM25_PATH = "chroma_db/bm25_index.pkl"

# tokenize the text
def tokenize(text: str) -> list[str]:
    return text.lower().split()

# Building bm25
def build_bm25(documents: list[str]) -> BM25Okapi:
    tokenized = [tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized)

# Saving bm25    
def save_bm25(documents: list[str], metadatas: list[dict] = None):
    """Build and persist BM25 index + raw docs + metadata."""
    index =  build_bm25(documents)
    os.makedirs(os.path.dirname(BM25_PATH),  exist_ok= True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": index, "docs": documents, "metadatas": metadatas or [{}] * len(documents)}, f)
    print(f"✅ BM25 index saved ({len(documents)} docs)")

# Loading bm25
def load_bm25() -> tuple[BM25Okapi, list[str], list[dict]]:
    """Load BM25 index,docs and metadatas form desk. """
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError("BM25 index not found. Run ingest first.")
    with open (BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["docs"], data.get("metadatas", [{}] * len(data["docs"]))

# BM25 Search
def bm25_search(query: str, top_k: int = 7) -> tuple[list[str], list[dict]]:
    """Returns docs and theri metadata."""
    index, docs, metadatas = load_bm25()
    tokens = tokenize(query)
    scores = index.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)
    top_docs =  [docs[i] for i in ranked_indices[:top_k]]
    top_metas = [metadatas[i] for i in ranked_indices[:top_k]]
    return top_docs, top_metas

# Rebuild BM25 after file deletion
def rebuild_bm25_without(filename: str):
    """Rebuild BM25 index excluding chunks from deleted files."""
    try:
        load_bm25()
    except FileNotFoundError:
        return  # Nothing to rebuild

    from rag.vectordb import get_collection
    collection = get_collection()

    # ✅ single call for both docs and metadatas
    remaining       = collection.get(include=["documents", "metadatas"])
    remaining_docs  = remaining["documents"]
    remaining_metas = remaining["metadatas"]

    if remaining_docs:
        save_bm25(remaining_docs, remaining_metas)
        print(f"✅ BM25 rebuilt with {len(remaining_docs)} chunks")
    else:
        if os.path.exists(BM25_PATH):
            os.remove(BM25_PATH)
        print("⚠️ No documents left in knowledge base")