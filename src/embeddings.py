from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

def embed_text(embedder, text):
    return embedder.encode(text).tolist()
