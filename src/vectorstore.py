import chromadb
from src.config import CHROMA_DIR

def get_vectorstore():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )

    return client, collection
