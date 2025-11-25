from src.embeddings import load_embedder, embed_text
from src.vectorstore import get_vectorstore
from src.config import DATA_DIR
import os
from tqdm import tqdm

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

def ingest_data():
    embedder = load_embedder()
    client, collection = get_vectorstore()

    for filename in tqdm(os.listdir(DATA_DIR)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(DATA_DIR, filename)
        text = open(path, "r", encoding="utf-8").read()
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            emb = embed_text(embedder, chunk)
            collection.add(
                ids=[f"{filename}-{i}"],
                documents=[chunk],
                metadatas=[{"source": filename}]
            )

    print("Ingestion completed.")

if __name__ == "__main__":
    ingest_data()
