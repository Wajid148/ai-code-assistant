from google import genai
import chromadb
import os
import time

client = genai.Client(api_key="GEMINI_API_KEY")

# CHANGED: PersistentClient saves to disk instead of memory
chroma_client = chromadb.PersistentClient(path="./chroma_db")

folder_path = r"C:\Users\hp\requests\src\requests"

def chunk_text(text, lines_per_chunk=50):
    lines = text.split("\n")
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i:i + lines_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def embed_with_retry(client, model, content, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(model=model, contents=content)
            return result
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(15)
    raise Exception("Failed after retries")

# Try to get existing collection, otherwise create + fill it
existing_collections = [c.name for c in chroma_client.list_collections()]

if "requests_lib_chunked" in existing_collections:
    print("Found existing database — skipping re-processing.\n")
    collection = chroma_client.get_collection(name="requests_lib_chunked")
else:
    print("No existing database found — processing files (one-time setup)...\n")
    collection = chroma_client.create_collection(name="requests_lib_chunked")

    files_to_process = [f for f in os.listdir(folder_path) if f.endswith(".py")]

    for filename in files_to_process:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            continue

        chunks = chunk_text(content)
        print(f"{filename} split into {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk{idx}"
            embedding_result = embed_with_retry(client, "gemini-embedding-001", chunk)
            embedding = embedding_result.embeddings[0].values

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"file": filename}]
            )
            print(f"Stored: {chunk_id}")
            time.sleep(5)

    print("\nDone processing. Database saved for next time.\n")

# --- Interactive question loop ---
print("Ask anything about the 'requests' library. Type 'exit' to quit.\n")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break

    question_embedding_result = embed_with_retry(client, "gemini-embedding-001", question)
    question_embedding = question_embedding_result.embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    print("Most relevant chunk(s):", results["ids"][0])

    relevant_text = "\n\n---\n\n".join(results["documents"][0])

    prompt = f"""Answer the question using ONLY the code below. Reference specific function or class names in your answer.

Code:
{relevant_text}

Question: {question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAnswer:", response.text, "\n")