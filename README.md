# STN10xScale.ai — Local RAG 

A lightweight, fully local Retrieval-Augmented Generation (RAG) system designed to answer HR and internal questions using documents stored in the repository. The system uses a local vector database for retrieval and a locally served LLM for generation — no external APIs or costs required.

## Side note
There is also a folder named "using_google_api" where I have used the google Gemini API key as asked, but it was giving a 504 downtime error, and i just couldn't figure out why this was happening. I did create the entire pipeline initially using Google's API key, but since that didn't work had to start working with Mistral 7B and I ran it locally on my machine using my laptop GPU.

## Key Components

- **Vector DB:** ChromaDB (local SQLite store)
- **Embeddings:** sentence-transformers (local embedding model)
- **LLM Server:** LM Studio serving a GGUF model (e.g. Mistral-7B-Instruct)
- **Data folder:** Put your documents in the `data/` directory (plain text files are supported)

## Features

- Fully offline retrieval + generation
- Document ingestion into Chroma vector store
- Query the knowledge base via CLI script

## Quickstart (Windows / PowerShell)

Prerequisites

- Python 3.10+ and `pip`
- Install package dependencies from `requirements.txt` (project root)
- LM Studio installed and a compatible GGUF model available locally

Install Python dependencies

```powershell
pip install -r requirements.txt
```

Start the local LLM (LM Studio)

1. Open LM Studio and load a GGUF model (recommended: Mistral-7B-Instruct).
2. In LM Studio Developer settings enable the local HTTP server.
3. Start the model and confirm the server is available at `http://localhost:1234/v1/chat/completions`.

Ingest documents into Chroma

```powershell
# builds/updates the vector DB from files in the `data/` folder
python -m src.ingest
```

Run a query

```powershell
# example: ask a question and set a role 
python -m src.main --q "Can interns work from home?" --role intern
```

## Configuration

- The main configuration is in `src/config.py` — adjust endpoints, model names, and vectorstore settings there.

## Project Structure

- `src/` — application source code
	- `ingest.py` — ingest documents into ChromaDB
	- `main.py` — CLI entrypoint to run queries
	- `vectorstore.py` — wrapper around ChromaDB operations
	- `embeddings.py` — embedding model logic
	- `retriever.py` — retrieval / ranking logic
	- `llm.py` — LLM request/response helpers
	- `utils.py`, `config.py`, `test_llm.py` — utilities and tests
- `data/` — place your documents here (e.g., `employee_handbook_v1.txt`)
- `chromadb/` — local ChromaDB files (SQLite + persisted vectors)

## Usage Notes

- Keep sensitive or private documents in the `data/` folder; they remain local.
- Re-run `python -m src.ingest` whenever you add/modify documents.
- If the LLM server address differs, update `src/config.py` or pass appropriate environment variables.

## Troubleshooting

- If you see connection errors to the LLM, verify LM Studio is running and the HTTP server is enabled.
- If embeddings fail, ensure the local embedding model is installed and accessible.

## Conflict Logic

Summary of how conflicts are (and can be) handled in this project:

- Current implementation (what's in `src/`):
	- During ingestion (`src/ingest.py`) each chunk is stored with a simple `{"source": filename}` metadata field.
	- Retrieval (`src/retriever.py`) is a pure similarity search over embeddings (no metadata filtering, no explicit re-ranking).
	- The RAG pipeline (`src/rag_pip.py`) concatenates the top retrieved passages into `CONTEXT` and sends a single prompt to the LLM that says to "Use ONLY the context below." The LLM is therefore responsible for resolving conflicts when composing the final answer.

- What that means in practice:
	- There is no automatic metadata-based filtering (e.g., `role: intern` vs `role: employee`) in the base `src/` pipeline — conflicting statements are resolved by the LLM using the retrieved text.
	- No explicit re-ranking step is performed before calling the LLM; retrieval relevance is entirely embedding-similarity driven.

- Example prompt snippet to encourage safe conflict resolution and citation-style answers:

```text
You are an HR assistant. Use ONLY the CONTEXT below to answer. If multiple context passages conflict, do the following:
1) Prefer documents that match the user's role (if provided).
2) Prefer documents with later dates for policy/updates.
3) Prefer more specific statements over generic ones.
4) If a decision is ambiguous, state the ambiguity and list the conflicting sources (include `source` metadata).

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER (include short citations like [source.txt]):
```

This pattern makes conflict-handling explicit and produces traceable answers.


## Cost Analysis (Gemini Flash inference cost estimate)

Below is a simple way to estimate inference cost if you replaced the local LLM with a hosted model such as Gemini Flash. Exact prices vary — check the latest Google Cloud / Gemini pricing — but the math below lets you plug in current unit costs.

- Formula:
	- tokens_per_query = context_tokens + prompt_tokens + response_tokens
	- daily_tokens = tokens_per_query * queries_per_day
	- cost_per_day = (daily_tokens / 1000) * price_per_1k_tokens

- Example assumptions and ranges (replace `price_per_1k_tokens` with the current Gemini Flash rate):
	- Example A (lean): context=1000 tokens, response=200 tokens → tokens_per_query ≈ 1200
		- For 5,000 queries/day → daily_tokens = 6,000,000 tokens (6,000k)
		- If `price_per_1k_tokens` = $0.0005 → cost/day ≈ $3 → cost/month ≈ $90
		- If `price_per_1k_tokens` = $0.005  → cost/day ≈ $30 → cost/month ≈ $900
	- Example B (heavy): context=3000 tokens, response=500 tokens → tokens_per_query ≈ 3500
		- For 5,000 queries/day → daily_tokens = 17,500,000 tokens (17,500k)
		- If `price_per_1k_tokens` = $0.0005 → cost/day ≈ $8.75 → cost/month ≈ $262.50
		- If `price_per_1k_tokens` = $0.005  → cost/day ≈ $87.50 → cost/month ≈ $2,625


