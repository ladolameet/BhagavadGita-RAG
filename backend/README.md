# Bhagavad Gita RAG Backend (FastAPI)

Deploy on Render:
1. Create a new Web Service
2. Connect this repo
3. Set root directory to /backend
4. Build command:
   pip install -r requirements.txt
5. Start command:
   uvicorn app:app --host 0.0.0.0 --port $PORT
6. Add Environment Variable:
   GOOGLE_API_KEY = your Gemini key
