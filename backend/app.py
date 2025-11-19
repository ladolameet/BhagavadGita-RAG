import os
import re
import pickle
import unicodedata
import json
import numpy as np
import PyPDF2
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================
PDF_FILE_DEFAULT = "Bhagavad-gita-Swami-BG-Narasingha.pdf"
DOCS_PICKLE = "docs_inmemory.pkl"
EMBS_NPY = "embs_inmemory.npy"  # must end with .npy

load_dotenv("google_key.env")
API_KEY = os.getenv("GOOGLE_API_KEY", "")

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.0-flash:generateContent"
)

TEMPERATURE = 0.15
MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CLEAN SANSKRIT (REMOVE JUNK)
# ============================================================
DEV_START = 0x0900
DEV_END = 0x097F


def is_devanagari(ch):
    cp = ord(ch)
    return DEV_START <= cp <= DEV_END


def clean_sanskrit(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    out = []

    for ch in text:
        cp = ord(ch)

        # remove Private Use Area junk
        if 0xE000 <= cp <= 0xF8FF:
            continue

        if is_devanagari(ch):
            out.append(ch)
            continue

        if ch.isdigit() or ch.isspace():
            out.append(ch)
            continue

        if ch in " ।॥,-—()[]?;:":
            out.append(ch)
            continue

    cleaned = re.sub(r"\s+", " ", "".join(out)).strip()
    return cleaned


# ============================================================
# VECTOR STORE
# ============================================================
class InMemoryVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.embeddings = np.zeros((0, dim), dtype=np.float32)
        self.metadatas = []
        self.texts = []

    def add(self, embs, metas, texts):
        embs = np.array(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        if embs.shape[1] != self.dim:
            raise ValueError("Embedding dimension mismatch")

        if self.embeddings.size == 0:
            self.embeddings = embs.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, embs])

        self.metadatas.extend(metas)
        self.texts.extend(texts)

    def search(self, q_emb, k=5):
        if self.embeddings.shape[0] == 0:
            return []

        # normalize rows safely
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        emb_norm = self.embeddings / norms
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        sims = emb_norm.dot(q_norm)
        top = sims.argsort()[::-1][:k]

        return [(float(sims[i]), self.metadatas[i], self.texts[i]) for i in top]

    def save(self):
        # save metadata and texts with pickle
        with open(DOCS_PICKLE, "wb") as f:
            pickle.dump({"metadatas": self.metadatas, "texts": self.texts}, f)

        # FIX: write .npy using an explicit file handle to avoid .tmp rename issues on Windows
        with open(EMBS_NPY, "wb") as f:
            np.save(f, self.embeddings)

    @classmethod
    def load(cls, dim):
        if not (os.path.exists(DOCS_PICKLE) and os.path.exists(EMBS_NPY)):
            return None
        try:
            data = pickle.load(open(DOCS_PICKLE, "rb"))
            with open(EMBS_NPY, "rb") as f:
                embs = np.load(f)
        except Exception:
            # if loading fails, return None so caller can rebuild index
            return None
        store = cls(dim)
        store.metadatas = data.get("metadatas", [])
        store.texts = data.get("texts", [])
        store.embeddings = embs
        return store


# ============================================================
# EXTRACT SHLOKAS
# ============================================================
DEV_RE = re.compile(r"[\u0900-\u097F]{3,}")


def extract_shlokas(text):
    lines = [clean_sanskrit(l) for l in text.splitlines() if l.strip()]
    out = []
    buf = []

    for ln in lines:
        if DEV_RE.search(ln):
            buf.append(ln)
        else:
            if buf:
                out.append(" ".join(buf))
                buf = []
    if buf:
        out.append(" ".join(buf))

    return out


# ============================================================
# CHUNK PDF
# ============================================================
def chunk_text(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=550,
        chunk_overlap=70,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [clean_sanskrit(c) for c in chunks if clean_sanskrit(c).strip()]


# ============================================================
# LOAD PDF
# ============================================================
def load_pdf(path):
    reader = PyPDF2.PdfReader(path)
    docs = []

    for pno, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        cleaned = clean_sanskrit(raw)

        if not cleaned.strip():
            continue

        shlokas = extract_shlokas(cleaned)
        chunks = chunk_text(cleaned)

        for idx, ch in enumerate(chunks, 1):
            docs.append({
                "page": pno,
                "chunk": idx,
                "text": ch,
                "shlokas": [s for s in shlokas if s in ch]
            })

    return docs


# ============================================================
# EMBEDDING MODEL
# ============================================================
print("Loading embedding:", MODEL_NAME)
try:
    sbert = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print("Failed to load SentenceTransformer:", e)
    sbert = None

EMB_DIM = sbert.get_sentence_embedding_dimension() if sbert is not None else 384


# ============================================================
# GLOBAL VECTOR STORE
# ============================================================
vector_store = None


# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
def startup():
    global vector_store

    saved = InMemoryVectorStore.load(EMB_DIM)
    if saved:
        vector_store = saved
        print("Loaded saved embeddings.")
        return

    if not os.path.exists(PDF_FILE_DEFAULT):
        print("PDF missing. Upload via /upload_pdf")
        vector_store = InMemoryVectorStore(EMB_DIM)
        return

    if sbert is None:
        print("Embedding model not available. Cannot index PDF.")
        vector_store = InMemoryVectorStore(EMB_DIM)
        return

    print("Indexing PDF first time...")
    docs = load_pdf(PDF_FILE_DEFAULT)

    if not docs:
        vector_store = InMemoryVectorStore(EMB_DIM)
        print("No text extracted from PDF.")
        return

    texts = [d["text"] for d in docs]
    try:
        embs = sbert.encode(texts, normalize_embeddings=True).astype(np.float32)
    except TypeError:
        # fallback if normalize_embeddings not supported
        embs = sbert.encode(texts)
        # normalize manually
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = (embs / norms).astype(np.float32)

    store = InMemoryVectorStore(EMB_DIM)
    metas = [{"page": d["page"], "chunk": d["chunk"], "shlokas": d["shlokas"]} for d in docs]

    store.add(embs, metas, texts)
    store.save()

    vector_store = store
    print("Indexing complete.")


# ============================================================
# GEMINI CALL
# ============================================================
def call_gemini(prompt):
    if not API_KEY:
        return "Missing Gemini API key."

    try:
        r = requests.post(
            f"{GEMINI_ENDPOINT}?key={API_KEY}",
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": TEMPERATURE}
            },
            timeout=25
        )
        data = r.json()
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
    except Exception as e:
        return f"Gemini error: {e}"


# ============================================================
# CHAT API
# ============================================================
class Query(BaseModel):
    query: str
    k: int = 5


@app.post("/chat")
async def chat(q: Query):
    global vector_store

    if vector_store is None or vector_store.embeddings.size == 0:
        return {"answer": "Index not ready.", "sources": []}

    if sbert is None:
        return {"answer": "Embedding model not loaded.", "sources": []}

    q_emb = sbert.encode([q.query], normalize_embeddings=True)[0]
    results = vector_store.search(q_emb, q.k)

    context = []
    sources = []

    for score, meta, txt in results:
        snippet = txt[:350].replace("\n", " ")
        context.append(f"Page {meta['page']} Chunk {meta['chunk']}: {snippet}")
        sources.append({
            "page": meta["page"],
            "chunk": meta["chunk"],
            "shlokas": meta.get("shlokas", []),
            "score": score
        })

    full_context = "\n".join(context)

    prompt = f"""
Use the Bhagavad Gita text.

Context:
{full_context}

Question:
{q.query}

Explain simply. No bullet points, no formatting, no lists. Plain easy language.
"""

    answer = call_gemini(prompt)

    return {"answer": answer, "sources": sources}


# ============================================================
# PDF UPLOAD
# ============================================================
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    path = "uploaded.pdf"
    with open(path, "wb") as f:
        f.write(await file.read())

    docs = load_pdf(path)

    if not docs:
        return {"ok": False, "message": "No extractable text in PDF."}

    texts = [d["text"] for d in docs]

    if sbert is None:
        return {"ok": False, "message": "Embedding model not loaded on server."}

    try:
        embs = sbert.encode(texts, normalize_embeddings=True)
    except TypeError:
        embs = sbert.encode(texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = (embs / norms).astype(np.float32)

    store = InMemoryVectorStore(EMB_DIM)
    metas = [{"page": d["page"], "chunk": d["chunk"], "shlokas": d["shlokas"]} for d in docs]

    store.add(embs, metas, texts)
    store.save()

    global vector_store
    vector_store = store

    return {"ok": True, "message": "Uploaded & indexed."}


# ============================================================
# STATUS
# ============================================================
@app.get("/status")
async def status():
    if not os.path.exists(DOCS_PICKLE) or not os.path.exists(EMBS_NPY):
        return {"indexed_chunks": 0, "embedding_dim": EMB_DIM}

    store = InMemoryVectorStore.load(EMB_DIM)
    if store is None:
        return {"indexed_chunks": 0, "embedding_dim": EMB_DIM}
    return {"indexed_chunks": len(store.texts), "embedding_dim": EMB_DIM}
