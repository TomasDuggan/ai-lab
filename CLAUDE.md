# CLAUDE.md

Contexto para Claude Code en este repo. Actualizar solo cuando algo cambie de verdad (stack, estructura, reglas) — no documentar cada feature.

## Qué es esto

Monorepo de un lab de experimentos de IA. Sin portfolio, sin auth, sin perfil — solo landing + rutas por experimento. No agregar estructura especulativa sin razón concreta.

## Stack (no reabrir sin pedido explícito)

- **Frontend**: Next.js (App Router) + TypeScript + Tailwind. Sin API routes propias — fetch directo al backend desde el browser.
- **Backend**: FastAPI (Python). Acá vive toda la lógica de IA.
- **Vector DB**: Chroma local/embebido.
- **LLM**: OpenRouter, modelos `:free`. Claude API solo para comparación puntual de calidad, nunca 24/7.
- **Embeddings**: sentence-transformers local u OpenRouter free tier.

## Reglas de costos

Todo gratis en desarrollo. Si una sugerencia implica costo recurrente, avisar explícitamente y dar alternativa gratuita primero.

## Cómo trabajar en este repo

- Priorizar explicar el razonamiento de decisiones de diseño (chunking, retrieval, arquitectura) antes que tirar código completo directo, salvo que se pida código directo.
- No explicar conceptos básicos de programación ni de Python/CS general — usuario nivel avanzado pero con no demasiada experiencia en python puntualmente.
- Sin cronogramas rígidos por fecha. Bloques de trabajo/hitos.

## Comandos

```bash
# backend
cd backend && source venv/bin/activate && uvicorn main:app --reload

# frontend
cd frontend && npm run dev
```