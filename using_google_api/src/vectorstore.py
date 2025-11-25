from src.path_setup import *

import chromadb
from src.config import CHROMA_DIR, get_data_files
from src.embeddings import get_google_embeddings
import os
import time
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
from src.utils import chunk_text

def load_documents():
    docs, metas, ids = [], [], []
    idx = 0
    for file in get_data_files():
        text = file.read_text()
        name = file.name.lower()
        if "intern" in name:
            meta_base = {"role": "intern", "type": "policy", "source": file.name}
        elif "manager" in name:
            meta_base = {"role": "employee", "type": "policy_update", "source": file.name}
        else:
            meta_base = {"role": "employee", "type": "policy", "source": file.name}
        for chunk in chunk_text(text):
            docs.append(chunk)
            metas.append(meta_base)
            ids.append(f"doc_{idx}")
            idx += 1
    return docs, metas, ids

def build_vectorstore():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if "company_policies" in [c.name for c in client.list_collections()]:
        client.delete_collection("company_policies")
    
    embeddings = get_google_embeddings()
    collection = client.create_collection(
        name="company_policies",
        metadata={"hnsw:space": "cosine"}
    )
    
    docs, metas, ids = load_documents()
    
    # Process in smaller batches with retry logic
    batch_size = 5  # Reduce batch size
    max_retries = 3
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        for attempt in range(max_retries):
            try:
                print(f"Processing batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1}...")
                vectors = embeddings.embed_documents(batch_docs)
                
                collection.add(
                    documents=batch_docs,
                    embeddings=vectors,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                print(f"✓ Batch {i//batch_size + 1} completed")
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"⚠ Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed after {max_retries} attempts")
                    raise
        
        # Small delay between batches to avoid rate limiting
        time.sleep(0.5)
    
    print("Vectorstore built successfully!")
    return collection