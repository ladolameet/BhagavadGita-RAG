from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 🔐 Load API Key
# ============================================================
load_dotenv("google_key.env")
API_KEY = os.getenv("GOOGLE_API_KEY")

ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.0-flash:generateContent"
)
TEMPERATURE = 0.5

# ============================================================
# 📘 Load Handbook Chunks
# ============================================================
df = pd.read_csv("gita_pypdf2.csv")  # MUST contain: chapter | verse | chunk

# create clean text for embeddings
docs = [
    f"Chapter {row['chapter']}, Verse {row['verse']}: {row['text']}"
    for _, row in df.iterrows()
]

# ============================================================
# 🔍 Build Embeddings + FAISS Index
# ============================================================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(docs, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))
print("📚 FAISS Index Ready:", index.ntotal, "Gita chunks loaded")


# ============================================================
# 🚀 FastAPI Setup + CORS (Frontend Compatible)
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.options("/chat")
async def chat_options():
    return {"status": "ok"}

# ============================================================
# 📩 Input Model
# ============================================================
class Query(BaseModel):
    query: str

# ============================================================
# 🤖 RAG + Gemini Function
# ============================================================
def rag_query_gemini(query, k=4):
    # 1️⃣ Encode user query
    q_emb = embed_model.encode([query], normalize_embeddings=True)

    # 2️⃣ Search FAISS
    scores, idxs = index.search(np.array(q_emb), k)
    retrieved_chunks = [docs[i] for i in idxs[0]]

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are an AI assistant specialized in the Bhagavad Gita.

Use the Bhagavad Gita context below to answer the user's question.

Rules:
1. If the Gita context contains the answer, explain it clearly.
2. If the context is not enough, answer using general spiritual, philosophical,
   or historical knowledge of the Gita.
3. Give simple, easy-to-understand explanations.
4. Mention relevant chapter and verse numbers when helpful.
5. Never say "I cannot answer." Always give a meaningful and helpful reply.

Bhagavad Gita Context:
{context}

User Question:
{query}

Give a clear, easy explanation.
"""


    # 4️⃣ Call Gemini API
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": TEMPERATURE}
    }

    url = f"{ENDPOINT}?key={API_KEY}"
    response = requests.post(url, headers=headers, json=body)

    if not response.ok:
        return "⚠ API Error", retrieved_chunks

    data = response.json()

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        answer = "⚠ No valid response from Gemini."

    return answer, retrieved_chunks

# ============================================================
# 🌐 CHAT Endpoint (POST + GET)
# ============================================================
@app.api_route("/chat", methods=["GET", "POST"])
async def chat_api(request: Request):
    # GET request support
    if request.method == "GET":
        query = request.query_params.get("query", "")
        if not query:
            return {"error": "Query parameter is missing."}
    # POST request support
    else:
        body = await request.json()
        query = body.get("query", "")

    answer, retrieved = rag_query_gemini(query)
    return {"answer": answer, "retrieved": retrieved}
