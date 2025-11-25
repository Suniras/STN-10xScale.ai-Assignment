from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY
import time


def get_google_embeddings(timeout: int = 120):
    """Return a Google Generative AI embeddings client with configurable timeout."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document",
        request_options={"timeout": timeout},
    )


def get_hf_embeddings():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts, use_google=True):
    """Backward-compatible embed helper (simple call).

    Prefer `embed_texts_with_retries` for production robustness.
    """
    if use_google:
        model = get_google_embeddings()
        return model.embed_documents(texts)
    else:
        model = get_hf_embeddings()
        return model.encode(texts).tolist()


def embed_texts_with_retries(
    texts,
    use_google=True,
    retries: int = 3,
    backoff_factor: int = 2,
    timeout: int = 120,
):
    """Try to embed `texts` using Google with retries and exponential backoff.

    If Google repeatedly fails, fall back to a local SentenceTransformer embedding.
    Returns a list of vectors.
    """
    if not use_google:
        return get_hf_embeddings().encode(texts).tolist()

    attempt = 0
    wait = 1
    last_exc = None
    while attempt <= retries:
        try:
            model = get_google_embeddings(timeout=timeout)
            return model.embed_documents(texts)
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt > retries:
                # Final fallback to local HF embeddings
                return get_hf_embeddings().encode(texts).tolist()
            time.sleep(wait)
            wait *= backoff_factor

