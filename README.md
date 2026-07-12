# AI Code Assistant

An AI-powered tool that reads a real Python codebase and answers questions about it using Retrieval-Augmented Generation (RAG) and tool-based reasoning.

**Live demo:** https://web-production-fc38c4.up.railway.app

## What it does

- Reads all Python files from a codebase (currently tested on the `requests` library)
- Splits large files into smaller chunks for accurate retrieval
- Converts each chunk into embeddings using Google's Gemini API
- Stores embeddings in a vector database (ChromaDB)
- Decides whether to answer via direct file lookup or semantic search, depending on the question
- When asked a question, retrieves only the most relevant code chunks
- Sends those chunks to Gemini to generate an accurate, code-grounded answer

## Example questions it can answer

- How does authentication work in this library?
- What happens if a request gets redirected too many times?
- How does this library decide whether to reuse a connection?
- What does exceptions.py contain?

## Tech stack

- Python
- Google Gemini API (`google-genai`)
- ChromaDB (vector database)
- FastAPI (web interface)
- Deployed on Railway

## How it works

1. **Chunking** — large files are split into ~50 line pieces so retrieval finds precise, relevant code instead of entire files
2. **Embeddings** — each chunk is converted into a numerical representation of its meaning
3. **Tool selection** — the system checks whether a question names a specific file; if so, it reads that file directly instead of searching
4. **Retrieval** — otherwise, it finds the top 3 most relevant chunks by comparing meaning, not just keywords
5. **Generation** — Gemini answers the question using only the retrieved or read content, citing real function and class names

## Status

Fully working and deployed. Next steps: adding a second tool (keyword search), expanding to additional codebases, and building a multi-agent research system as a follow-up project.

## Local setup

1. Install dependencies: `pip install google-genai chromadb fastapi uvicorn`
2. Set your API key: `set GEMINI_API_KEY=your-key-here` (Windows) or `export GEMINI_API_KEY=your-key-here` (Mac/Linux)
3. Run the terminal version: `python test.py`
4. Or run the web version: `uvicorn app:app --reload` then open `http://127.0.0.1:8000`

## Deployment

Deployed on [Railway](https://railway.app) with `GEMINI_API_KEY` set as an environment variable.
