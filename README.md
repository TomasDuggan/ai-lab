# lab-ia

Laboratorio de experimentos de IA (RAG, agentes, LLMOps). Cada experimento es una ruta nueva en el frontend, con su lógica real viviendo en el backend.

## Estructura

```
lab-ia/
├── frontend/   → Next.js + Tailwind (landing + rutas por experimento)
├── backend/    → FastAPI (lógica de IA: RAG, agentes, LLM calls)
```

Sin capa intermedia: el frontend pega directo al backend desde el browser. Sin API routes de Next.js por ahora.

## Levantar local

**Backend**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```
→ http://localhost:8000/docs

**Frontend**
```bash
cd frontend
npm run dev
```
→ http://localhost:3000

## Experimentos

| Ruta   | Estado      | Descripción |
|--------|-------------|-------------|
| `/rag` | pendiente   | — |

## Stack

Next.js + Tailwind · FastAPI · Chroma (vector DB local) · OpenRouter (LLM, modelos `:free`) · sentence-transformers (embeddings locales)

## Deploy

Frontend → Vercel · Backend → Render/Railway (free tier)