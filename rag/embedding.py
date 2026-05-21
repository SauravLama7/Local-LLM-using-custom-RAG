from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    model = get_model()
    return model.encode(texts, convert_to_numpy = True, normalize_embeddings=True)
