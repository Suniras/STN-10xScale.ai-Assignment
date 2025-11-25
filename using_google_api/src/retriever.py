from src.path_setup import *

import chromadb
from src.config import CHROMA_DIR
from src.embeddings import get_google_embeddings

def retrieve(query: str, role: str = "employee", top_k: int = 5):
    """
    Retrieve relevant documents with role-based filtering
    
    Args:
        query: User's question
        role: User role (intern/employee)
        top_k: Number of documents to retrieve
    
    Returns:
        List of tuples: [(document_text, metadata), ...]
    """
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("company_policies")
    
    # Get query embedding
    embeddings = get_google_embeddings()
    query_vector = embeddings.embed_query(query)
    
    # Strategy: Retrieve role-specific docs first, then general docs
    retrieved_docs = []
    
    # 1. Get role-specific documents (highest priority)
    if role == "intern":
        intern_results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"role": "intern"}
        )
        
        if intern_results['documents'][0]:
            for doc, meta in zip(intern_results['documents'][0], 
                                intern_results['metadatas'][0]):
                retrieved_docs.append((doc, meta))
    
    # 2. Get general employee policies (for context)
    employee_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"role": "employee"}
    )
    
    if employee_results['documents'][0]:
        for doc, meta in zip(employee_results['documents'][0], 
                            employee_results['metadatas'][0]):
            retrieved_docs.append((doc, meta))
    
    # Remove duplicates while preserving order (role-specific first)
    seen = set()
    unique_docs = []
    for doc, meta in retrieved_docs:
        doc_id = (doc, meta['source'])
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append((doc, meta))
    
    # Limit to top_k results
    return unique_docs[:top_k]