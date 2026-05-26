from sentence_transformers import CrossEncoder
import streamlit as st

@st.cache_resource
def get_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, documents, top_k=5):
    if not documents:
        return []
    
    model = get_reranker() 
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]