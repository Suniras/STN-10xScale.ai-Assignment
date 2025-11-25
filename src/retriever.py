from src.embeddings import load_embedder, embed_text
from src.vectorstore import get_vectorstore

def retrieve_docs(query, n=5):
    embedder = load_embedder()
    q_emb = embed_text(embedder, query)

    client, collection = get_vectorstore()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n
    )

    docs = results["documents"][0]
    return docs
