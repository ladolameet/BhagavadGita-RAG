from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import pickle

# ============================================================
# 🚀 Create FastAPI App FIRST (very important!)
# ============================================================
app = FastAPI()

# ============================================================
# 🔐 API Key
# ============================================================
load_dotenv("google_key.env")
API_KEY = os.getenv("GOOGLE_API_KEY")

ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.0-flash:generateContent"
)
TEMPERATURE = 0.5

# ============================================================
# Global Variables
# ============================================================
docs = []
index = None

# ============================================================
# ⚡ Embeddings (CPU Mode — Safe)
# ============================================================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("⚡ Running embeddings on CPU")


# ============================================================
# 🧩 Smart Chunking Function
# ============================================================
def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])


# ============================================================
# 📘 Load PDF with chunking
# ============================================================
def load_pdf_to_docs(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    all_docs = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        for i, chunk in enumerate(chunk_text(text)):
            all_docs.append({
                "page": page_number,
                "chunk": i + 1,
                "text": chunk
            })

    return all_docs


# ============================================================
# 💾 Save & Load Embeddings
# ============================================================
def save_index(docs, index):
    pickle.dump(docs, open("docs.pkl", "wb"))
    faiss.write_index(index, "faiss.index")


def load_index_if_exists():
    if os.path.exists("docs.pkl") and os.path.exists("faiss.index"):
        print("⚡ Loading saved FAISS index…")
        docs = pickle.load(open("docs.pkl", "rb"))
        index = faiss.read_index("faiss.index")
        return docs, index
    return None, None


# ============================================================
# 🚀 Startup Event — Build / Load FAISS
# ============================================================
@app.on_event("startup")
def build_index():
    global docs, index

    # 1️⃣ Try to load cached index
    docs_cache, index_cache = load_index_if_exists()
    if docs_cache and index_cache:
        docs = docs_cache
        index = index_cache
        print("🔥 Loaded embeddings from disk")
        return    # <-- THIS IS IMPORTANT!


# ============================================================
# 🚀 CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# 📩 Input Model
# ============================================================
class Query(BaseModel):
    query: str


# ============================================================
# 🔍 RAG Query Function
# ============================================================
def rag_query_gemini(query, k=5):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb), k)

    retrieved = [docs[i] for i in idxs[0]]

    context = "\n".join(
        [f"Page {d['page']} | Chunk {d['chunk']}: {d['text']}" for d in retrieved]
    )

    prompt = f"""
Use the Bhagavad Gita PDF as your reference.

Context from PDF:
{context}

User Question:
{query}

Answer in a simple, easy-to-understand way.
Respond ONLY in plain text. 
Do NOT use **bold**, *, _, #, lists, or any markdown formatting.
Write the answer in a simple and easy-to-understand way.
"""

    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": TEMPERATURE}
    }

    response = requests.post(f"{ENDPOINT}?key={API_KEY}",
                             headers=headers,
                             json=body)

    try:
        answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        answer = "⚠ Error generating response."

    return answer, retrieved


# ============================================================
# 🌐 CHAT Endpoint
# ============================================================
@app.api_route("/chat", methods=["POST", "GET"])
async def chat_api(request: Request):
    if request.method == "GET":
        query = request.query_params.get("query", "")
    else:
        body = await request.json()
        query = body.get("query", "")

    answer, retrieved = rag_query_gemini(query)

    return {
        "answer": answer,
        "sources": [
            {
                "page": d["page"],
                "chunk": d["chunk"],
                "text": d["text"][:180] + "..."
            }
            for d in retrieved
        ]
    }


# ============================================================
# 📤 Upload PDF Endpoint
# ============================================================
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    filename = "uploaded.pdf"
    with open(filename, "wb") as f:
        f.write(await file.read())

    return {"message": "PDF uploaded successfully. Restart server to re-index."}
