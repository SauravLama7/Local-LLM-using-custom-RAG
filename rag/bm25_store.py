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
def save_bm25(documents: list[str]):
    """Build and persist BM25 index + raw docs."""
    index =  build_bm25(documents)
    os.makedirs(os.path.dirname(BM25_PATH),  exist_ok= True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": index, "docs": documents}, f)
    print(f"✅ BM25 index saved ({len(documents)} docs)")

# Loading bm25
def load_bm25() -> tuple[BM25Okapi, list[str]]:
    """Load BM25 index and docs form desk. """
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError("BM25 index not found. Run ingest first.")
    with open (BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["docs"]

# BM25 Search
def bm25_search(query: str, top_k: int = 7) -> list[str]:
    index, docs = load_bm25()
    tokens = tokenize(query)
    scores = index.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)
    return [docs[i] for i in ranked_indices[:top_k]]
