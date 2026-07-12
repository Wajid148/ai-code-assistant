from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
import chromadb
import os

app = FastAPI()

# Read Gemini API key safely
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing")

print("Gemini API key configured:", True)

client = genai.Client(api_key=api_key)

# Create or open ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="requests_lib_chunked"
)

print("Chroma collection count:", collection.count())

# Railway-compatible source folder
folder_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "requests_source"
)

if os.path.isdir(folder_path):
    all_filenames = [
        filename
        for filename in os.listdir(folder_path)
        if filename.endswith(".py")
    ]
else:
    all_filenames = []
    print(f"Warning: source folder not found: {folder_path}")


def read_file_tool(filename: str):
    filepath = os.path.join(folder_path, filename)

    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    return None


def detect_filename_in_question(question: str, filenames: list[str]):
    question_lower = question.lower()

    for filename in filenames:
        if filename.lower() in question_lower:
            return filename

    return None


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>AI Code Assistant</title>
    </head>

    <body style="font-family: sans-serif; max-width: 700px; margin: 40px auto;">
        <h2>Ask about the 'requests' library</h2>

        <input
            id="q"
            type="text"
            style="width:100%; padding:10px; font-size:16px;"
            placeholder="Type your question..."
        >

        <button
            onclick="ask()"
            style="margin-top:10px; padding:10px 20px;"
        >
            Ask
        </button>

        <div
            id="answer"
            style="margin-top:20px; white-space:pre-wrap; font-size:15px;"
        ></div>

        <script>
        async function ask() {
            const question = document.getElementById('q').value.trim();
            const answerBox = document.getElementById('answer');

            if (!question) {
                answerBox.innerText = "Please enter a question.";
                return;
            }

            answerBox.innerText = "Thinking...";

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        question: question
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    answerBox.innerText =
                        data.detail || "An error occurred.";
                    return;
                }

                answerBox.innerText = data.answer;
            } catch (error) {
                answerBox.innerText =
                    "Unable to contact the server.";
            }
        }
        </script>
    </body>
    </html>
    """


@app.post("/ask")
def ask(q: Question):
    question = q.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    named_file = detect_filename_in_question(
        question,
        all_filenames
    )

    if named_file:
        relevant_text = read_file_tool(named_file)

        if not relevant_text:
            raise HTTPException(
                status_code=404,
                detail="The requested source file could not be read."
            )

    else:
        if collection.count() == 0:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The ChromaDB collection exists but contains no "
                    "documents. Upload or generate the embeddings first."
                )
            )

        embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )

        question_embedding = (
            embedding_result.embeddings[0].values
        )

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )

        documents = results.get("documents", [[]])[0]

        if not documents:
            raise HTTPException(
                status_code=404,
                                detail="No relevant source code was found."
            )

        relevant_text = "\n\n---\n\n".join(documents)

    prompt = f"""
Answer the question using ONLY the code below.
Reference specific function or class names in your answer.
If the answer is not present in the code, say that clearly.

Code:
{relevant_text}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "answer": response.text
    }