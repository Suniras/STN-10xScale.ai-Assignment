from src.config import GOOGLE_API_KEY
from src.embeddings import embed_texts_with_retries
import time

print("Testing configuration...")
print(f"✓ API Key loaded: {GOOGLE_API_KEY[:20]}...")

print("\nTesting embeddings with single text (with retries + fallback)...")
try:
    start = time.time()
    result = embed_texts_with_retries(["Hello world"], use_google=True)
    elapsed = time.time() - start
    print(f"✓ Embedding successful!")
    print(f"  Vector dimension: {len(result[0])}")
    print(f"  Time taken: {elapsed:.2f}s")
except Exception as e:
    print(f"✗ Error: {e}")