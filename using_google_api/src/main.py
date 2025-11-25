from src.path_setup import *

import sys, os
print(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore import build_vectorstore
from src.rag_pip import run_rag

def main():
    print("Building vectorstore...")
    build_vectorstore()
    
    print("\n" + "="*60)
    print("TESTING RAG SYSTEM")
    print("="*60)
    
    query = "I just joined as a new intern. Can I work from home?"
    print(f"\nQuery: {query}")
    print("\nProcessing...")
    
    answer, sources = run_rag(query, role="intern")
    
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
    
    print("\n" + "="*60)
    print("SOURCES USED:")
    print("="*60)
    for s in sources:
        print(f"  • {s}")
    print("="*60)

if __name__ == "__main__":
    main()