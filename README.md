# AI Code Assistant

An AI-powered tool that reads a real Python codebase and answers questions about it using Retrieval-Augmented Generation (RAG).

## What it does

- Reads all Python files from a codebase (currently tested on the requests library)
- Splits large files into smaller chunks for accurate retrieval
- Converts each chunk into embeddings using Google's Gemini API
- Stores embeddings in a local vector database (ChromaDB)
- When asked a question, retrieves only the most relevant code chunks
- Sends those chunks to Gemini to generate an accurate, code-grounded answer

## Example questions it can answer

- How does authentication work in this library?
- What happens if a request gets redirected too many times?
- How does this library decide whether to reuse a connection?

## Tech stack

- Python
- Google Gemini API (google-genai)
- ChromaDB (vector database)

## How it works

1. Chunking — large files are split into ~50 line pieces so retrieval finds precise, relevant code instead of entire files
2. Embeddings — each chunk is converted into a numerical representation of its meaning
3. Retrieval — when a question is asked, the system finds the top 3 most relevant chunks by comparing meaning, not just keywords
4. Generation — Gemini answers the question using only those retrieved chunks, citing real function and class names

## Status

Work in progress — next steps include adding tool-use (agent capabilities) and deploying as a live web app.

## Setup

1. Install dependencies: pip install google-genai chromadb
2. Set your API key: set GEMINI_API_KEY=your-key-here (Windows) or export GEMINI_API_KEY=your-key-here (Mac/Linux)
3. Run: python test.py
