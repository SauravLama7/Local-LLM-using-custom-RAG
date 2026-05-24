from collections import defaultdict

# Fusing BM25 and Vector Embedding
def reciprocal_rank_fusion(
        vector_docs: list[str],
        bm25_docs: list[str],
        k: int = 60
) -> list[str]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Score = sum of 1 / (k + rank) across both lists.
    """
    scores = defaultdict(float)

    for rank, doc in enumerate(vector_docs, start = 1):
        scores[doc] += 1 / (k + rank)

    for rank, doc in enumerate(bm25_docs, start = 1):
        scores[doc] += 1 / (k + rank)

    # Sort by fused score desending
    merged = sorted(scores.keys(), key = lambda d: scores[d], reverse = True)
    return merged