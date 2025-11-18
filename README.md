# Bhagavad Gita RAG Frontend (HTML + JS)

This is the frontend UI for the Bhagavad Gita AI Guide.

## How to Deploy on Vercel

1. Go to https://vercel.com
2. Create a new project.
3. Import this GitHub repo.
4. In settings, set:
   - Project Root Directory: frontend
   - Framework Preset: "Other"

5. Deploy.

## IMPORTANT
After backend is deployed on Render, update API URL inside script.js:

const API_URL = "https://your-backend-url.onrender.com/chat";
