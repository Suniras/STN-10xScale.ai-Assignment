import argparse
from src.rag_pip import rag_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, required=True)
    args = parser.parse_args()

    answer = rag_pipeline(args.q)
    print("\n=== ANSWER ===\n")
    print(answer)

if __name__ == "__main__":
    main()
