# lab-ia

AI experiments lab (RAG, agents, LLMOps). Each experiment is a new route in the frontend, with its real logic living in the backend.

## Structure

```
lab-ia/
├── frontend/   → Next.js + Tailwind (landing + routes per experiment)
├── backend/    → FastAPI (AI logic: RAG, agents, LLM calls)
```

No middle layer: the frontend hits the backend directly from the browser. No Next.js API routes for now.

## Running locally

**Backend**
```bash
cd backend
source venv/Scripts/activate
uvicorn main:app --reload
```
→ http://localhost:8000/docs

**Frontend**
```bash
cd frontend
npm run dev
```
→ http://localhost:3000

## Experiments

| Route  | Status    | Description |
|--------|-----------|-------------|
| `/rag` | pending   | — |

## Stack

Next.js + Tailwind · FastAPI · Chroma (local vector DB) · OpenRouter (LLM, `:free` models) · sentence-transformers (local embeddings)

## Deploy

Frontend → Vercel · Backend → Render/Railway (free tier)