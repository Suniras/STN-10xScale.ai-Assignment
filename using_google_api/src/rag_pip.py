from src.path_setup import *

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve
from src.llm import generate_answer


def run_rag(query, role="employee"):
    retrieved = retrieve(query, role=role)

    context_chunks = []
    sources = []

    for doc, meta in retrieved:
        context_chunks.append(doc)
        sources.append(meta.get("source"))

    context = "\n\n---\n\n".join(context_chunks)
    answer = generate_answer(context, query)

    return answer, sources
