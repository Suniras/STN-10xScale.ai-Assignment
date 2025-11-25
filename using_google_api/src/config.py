import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

# Alias for compatibility
GEMINI_API_KEY = GOOGLE_API_KEY

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chromadb"

# Updated model names for Gemini 2.0
GEMINI_MODEL = "gemini-2.0-flash-exp"  # or "gemini-1.5-flash" if 2.0 not available
EMBEDDING_MODEL = "models/text-embedding-004"

def get_data_files():
    return list(DATA_DIR.glob("*.txt"))