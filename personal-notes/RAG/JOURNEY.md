# Project journey - decisions above the pipeline level

Covers project-level decisions that don't belong to any single technical
stage (chunking, embeddings, Chroma, generation) - architecture,
workflow, and domain choices made while building the Steam RAG.

## Architecture

- **Monorepo, frontend and backend fully decoupled**: `/frontend`
  (Next.js) never calls OpenRouter or Chroma directly - all AI/retrieval
  logic lives in the FastAPI backend, and the frontend only talks to it
  over HTTP. Deliberate: the AI service can be unavailable/redeployed
  without breaking the frontend's ability to at least render, and each
  side can be deployed independently (Vercel + Render/Railway) without
  coupling their release cycles.

- **Setup (offline) vs runtime (per-request) kept as two distinct
  phases**: ingestion, chunking, embedding generation, and inserting into
  Chroma are all run manually as scripts, ahead of time - never triggered
  by an HTTP request. FastAPI, at runtime, only *reads* from an
  already-populated Chroma folder. Keeping these separate means a slow,
  occasionally-run process (re-embedding a corpus) is never accidentally
  in the critical path of a live request.

- **Core pipeline functions written as plain, dependency-injected Python,
  not FastAPI-coupled from the start**: `retrieve()` takes `model` and
  `collection` as parameters rather than reading them from global state
  or FastAPI's `Depends()` directly. This meant the entire retrieval and
  generation pipeline could be built, tested, and debugged with plain
  scripts - no server needed to be running to iterate - and only wired
  into FastAPI's dependency injection (`Depends()`, loaded once at
  startup via `lifespan`) as the very last step.

## Evaluation infrastructure

- **Built a manual eval harness (JSONL history, recall/MRR scoring)
  instead of reaching for an eval framework**: chosen for the same reason
  as the manual deduplication logic - full understanding of what's being
  measured, at a scale (4-10 eval cases) where a framework would add
  more overhead than value. Each run appends a labeled entry
  (`{timestamp, label, config, avg_recall, per-case detail}`) to a
  JSONL file rather than overwriting - a full run history was considered
  more valuable than a single "current" score, specifically to make
  before/after comparisons possible days or weeks apart.

- **Never trusted a single number to represent a change**: every config
  change (chunk_size, dedup, embedding model) was evaluated against the
  *same* eval set before drawing any conclusion - including catching two
  separate points where the temptation was to react to raw manual
  testing (4 hand-picked queries) instead of waiting for the harness.
  Both times, the harness result contradicted or refined the initial
  impression (chunk_size seeming clearly responsible for bad results,
  when the real lever turned out to be the embedding model).

## Domain choice

- **Chose Steam game descriptions as the first RAG's domain
  deliberately for its low stakes, not its realism**: the goal was to
  learn the mechanics of chunking/embeddings/retrieval/generation/
  evaluation hands-on, not to solve a real information-retrieval problem.
  Public, well-known, low-consequence data made debugging and evaluation
  easier to reason about (existing knowledge of games served as an
  intuition check on whether retrieval results made sense).

- **Concluded, with evidence, that this domain doesn't actually justify
  RAG in a real product**: a base LLM already knows popular games well
  from training data - in the worst observed case (Garry's Mod
  outranking Counter-Strike 2), RAG measurably made an answer worse than
  an ungrounded LLM would likely have given. This wasn't assumed
  up front; it was a conclusion reached only after building and
  evaluating the full pipeline. Directly motivated building a second RAG
  on a domain that actually meets the criteria where RAG earns its
  complexity: private data the LLM cannot already know (this project's
  own notes/decisions).

## Cost policy

- **Zero recurring cost enforced throughout, no exceptions taken**: local
  embeddings (sentence-transformers), local vector store (Chroma
  embedded), free-tier LLM calls (OpenRouter `:free` / `openrouter/free`)
  for all development and iteration. The only paid API considered
  (Claude, for LLM-as-judge evaluation) was scoped explicitly as
  pay-per-use for a specific future evaluation task, never as always-on
  infrastructure - and wasn't used yet at the time of writing this.

## Working style with the AI pair (Claude)

- **Retrieval logic corrections were driven by the human questioning
  results, not by trusting AI-suggested code by default**: several real
  bugs were caught this way rather than by Claude proactively flagging
  them - noticing Hollow Knight's absence from a hand-picked expected
  list, questioning whether `chunk_overlap` was actually visible in the
  output data, and pushing back on an over-eager "never do X" phrasing
  in a notes draft that overstated an untested claim. Each instance
  improved either the code or the accuracy of what got documented.