from src.retriever import retrieve_docs
from src.llm import generate_answer

def rag_pipeline(query):
    docs = retrieve_docs(query)

    context = "\n\n".join(docs)

    prompt = f"""
You are an HR assistant. Use ONLY the CONTEXT below to answer. If multiple context passages conflict, do the following:
1) Prefer documents that match the user's role (if provided).
2) Prefer documents with later dates for policy/updates.
3) Prefer more specific statements over generic ones.
4) If a decision is ambiguous, state the ambiguity and list the conflicting sources (include `source` metadata).

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    return generate_answer(prompt)
