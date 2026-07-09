from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
import chromadb
import os

app = FastAPI()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="requests_lib_chunked")

folder_path = r"C:\Users\hp\requests\src\requests"
all_filenames = [f for f in os.listdir(folder_path) if f.endswith(".py")]

def read_file_tool(filename):
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return None

def detect_filename_in_question(question, filenames):
    for filename in filenames:
        if filename.lower() in question.lower():
            return filename
    return None

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>AI Code Assistant</title></head>
    <body style="font-family: sans-serif; max-width: 700px; margin: 40px auto;">
        <h2>Ask about the 'requests' library</h2>
        <input id="q" type="text" style="width:100%; padding:10px; font-size:16px;" placeholder="Type your question...">
        <button onclick="ask()" style="margin-top:10px; padding:10px 20px;">Ask</button>
        <div id="answer" style="margin-top:20px; white-space: pre-wrap; font-size:15px;"></div>

        <script>
        async function ask() {
            const question = document.getElementById('q').value;
            document.getElementById('answer').innerText = "Thinking...";
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            });
            const data = await response.json();
            document.getElementById('answer').innerText = data.answer;
        }
        </script>
    </body>
    </html>
    """

@app.post("/ask")
def ask(q: Question):
    question = q.question
    named_file = detect_filename_in_question(question, all_filenames)

    if named_file:
        relevant_text = read_file_tool(named_file)
    else:
        embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )
        question_embedding = embedding_result.embeddings[0].values
        results = collection.query(query_embeddings=[question_embedding], n_results=3)
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

    return {"answer": response.text}