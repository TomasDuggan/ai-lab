# RAG (Retrieval-Augmented Generation)

## What it is

RAG is the pattern of giving an LLM external context (retrieved from a
vector store) so it can answer using information it wasn't trained on,
instead of relying only on its internal/general knowledge.

Two separate stages, each with its own failure modes:

1. **Retrieval**: query -> embedding -> vector search -> top-k chunks.
   Can fail even if generation is perfect (garbage in).
2. **Generation**: the LLM receives those chunks as context and produces
   an answer. Can fail even if retrieval is perfect (LLM ignores or
   misuses good context).

Debugging a bad RAG answer means figuring out **which** stage failed
first - printing the retrieved chunks (not just the final answer) is
the only way to tell them apart.

## The augmented prompt pattern

[SYSTEM PROMPT: behavior instructions]
+
[CONTEXT: retrieved chunks, inserted as text]
+
[USER QUESTION]
->
LLM generates an answer grounded in the context

Two design decisions that matter here, not just implementation detail:

- **"Answer using ONLY the provided context"** - without this instruction,
  the model can blend its own general knowledge with the retrieved
  corpus, and you lose control over where the answer actually comes
  from. This is the basis of what gets measured later as "faithfulness"
  in evaluation.
- **"If the context isn't enough, say so"** - without this, a RAG can
  force an answer even when retrieval gave it nothing useful, producing
  hallucination that looks grounded (worse than an obvious hallucination,
  because it appears legitimate).

## Message format (OpenAI-compatible APIs, e.g. OpenRouter)

Not a single string - a list of role-tagged messages:

```python
messages = [
    {"role": "system", "content": "You are an assistant that..."},
    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
]
```

Separating `system` (how to behave) from `user` (what data + question)
gives the model a clearer signal than dumping everything into one block.

## Building context from retrieved chunks

This part is application logic, not something a library gives you:

```python
def build_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for chunk in retrieved_chunks:
        parts.append(f"[{chunk['name']}] ({', '.join(chunk['genres'])})\n{chunk['chunk']}")
    return "\n\n".join(parts)
```

## Failure modes observed in practice (not just theory)

RAG quality depends entirely on retrieval quality - a perfect prompt and
a well-behaved LLM can't compensate for chunks that shouldn't have been
retrieved in the first place. Two concrete causes found in this project:

1. **Chunk dilutes into a "broad spectrum" topic.** A chunk covering many
   loosely-related sub-topics (e.g. a sandbox game whose text also
   mentions its multiplayer modes, murder-mystery mode, "thousands of
   players online") ends up with an embedding that sits in a vague
   middle ground - it can rank decently against many unrelated queries
   simply because it touches a bit of everything. Smaller, more focused
   chunks reduce (but don't eliminate) this.

2. **A chunk names other, unrelated games/entities directly in its
   text.** E.g. a sandbox game's chunk mentioning "requires
   Counter-Strike: Source" pulls that chunk's embedding toward the
   semantic neighborhood of those named games, even though the game
   itself isn't actually similar to them. This is harder to fix - it's
   not a chunk-size problem, it's a content problem. Shrinking the chunk
   doesn't remove the mention; it can even isolate it into a purer,
   more "concentrated" chunk that matches even more strongly against
   the wrong queries.

3. **Generic/overlapping vocabulary across genres causes semantic
   confusion even without any bad chunking or cross-mentions.** E.g. a
   query about "relaxing farming/life sim" retrieved survival-genre
   games (The Forest, DayZ) ahead of the actual match (Stardew Valley),
   because survival games and life-sim games both use words like
   "build", "craft", "gather resources", "daily life" - the embedding
   model conflates the vocabulary overlap with topical similarity. This
   is the least "fixable" failure - there's no bad token to point at, it's
   a genuine limitation of how the embedding model reads meaning.

**Important, and reassuring:** the LLM generation step, when properly
instructed, is robust against bad context - it correctly excluded
irrelevant retrieved games (e.g. Garry's Mod, The Elder Scrolls Online)
even when they were present in the context, rather than blindly
recommending them. Retrieval quality and generation quality are
genuinely separate problems - good generation instructions don't fix
bad retrieval, but they do limit the damage.

---
