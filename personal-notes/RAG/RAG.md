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

## Diversity-aware retrieval (deduplication by parent document)

Known pattern in mature RAG frameworks - LangChain/LlamaIndex call this
"parent document retrieval," and the more general version (balancing
relevance against diversity across *all* results, not just same-document
duplicates) is **MMR (Maximal Marginal Relevance)**.

Mechanism: over-fetch more chunks than needed from the vector store (e.g.
`n_results * 4`), then keep only the best-scoring chunk per unique
source document before truncating to the actual `n_results`. Most vector
stores return results already sorted by relevance, so keeping the first
occurrence of each document is sufficient - no manual re-sorting needed.

```python
def retrieve(query: str, model, collection, n_results: int = 5) -> list[dict]:
    query_embedding = model.encode([query]).tolist()

    query_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results * 4  # over-fetch for dedup headroom
    )

    all_chunks = [...]  # flatten documents/metadatas/distances into dicts

    seen_docs = {}
    for chunk in all_chunks:
        if chunk["name"] not in seen_docs:
            seen_docs[chunk["name"]] = chunk  # first = best, already sorted

    return list(seen_docs.values())[:n_results]
```

Why over-fetch instead of dedup-then-truncate: deduping only within the
original `n_results` window would just leave you with fewer results
whenever duplicates occur - it doesn't give the dedup step room to
actually find N *distinct* good candidates. Over-fetching first, then
deduping, then truncating preserves the intended result count.

## Evaluation set - core concept

A fixed set of (query, expected results) cases, re-run identically after
any pipeline change, to get comparable scores over time instead of
eyeballing a handful of manual queries each time.

**Metrics:**
- **Recall@k**: of the expected results, how many appeared in the top-k?
  Doesn't penalize noise, only measures presence.
- **Reciprocal Rank**: position of the *first* expected hit (1/rank).
  Complements recall - a query can have recall=1.0 while its one correct
  answer sits buried near the bottom, which recall alone can't
  distinguish from a top-1 hit.

**Ground truth quality is the ceiling on what the evaluation can tell
you.** A narrow, "only the 2-3 most obvious answers" expected list
produces misleadingly optimistic scores - not because retrieval is good,
but because the bar for "correct" was too easy to clear. Valid answers
missing from the expected list can silently punish the metric if they
displace an expected item from a size-limited top-k, without ever being
credited as a real hit.

---

## Decisions made in this project

- **"Answer using ONLY context" + "say so if context is insufficient"
  system prompt, always**: both included from the first working version.
  Confirmed effective in practice - the LLM correctly excluded irrelevant
  retrieved games (e.g. Garry's Mod, The Elder Scrolls Online) even when
  present in context, rather than blindly recommending them. Generation
  quality and retrieval quality turned out to be genuinely separate
  problems in this project - good instructions limited the damage of bad
  retrieval, but never fixed it.

- **Manual deduplication by document instead of using a
  framework's built-in retriever**: implemented the over-fetch + keep-
  best-per-document pattern by hand rather than reaching for
  LangChain/LlamaIndex's parent-document retriever, for full control and
  understanding at this scale. This was the highest-impact fix of the
  entire retrieval pipeline (see table below) - chunk-level search
  otherwise let a single game's multiple chunks dominate the top-k,
  crowding out genuinely different, relevant games.

- **Rebuilt the evaluation set's `expected_games` from a full corpus
  review, not memory**: an early version used only 1-4 "obviously
  correct" games per query, hand-picked from memory. This produced
  misleadingly high recall (0.65) - not because retrieval was good, but
  because the bar was too narrow. Rewrote it by checking every one of the
  77 games in the corpus against each query. This dropped baseline
  recall to 0.42 on the *same* retrieval code - the system didn't get
  worse, the measurement got honest. Also caught a real bug in the
  process: "Apex Legends" was in an early expected list but was never
  actually ingested into the corpus - no retrieval change could ever
  have found it.

- **Chose `all-mpnet-base-v2` over `all-MiniLM-L6-v2` based on measured
  recall, not intuition**: see embeddings.md for the full comparison
  (0.504 -> 0.783 recall on the same eval set). Included here because it
  ended up being the single biggest lever in the whole RAG pipeline -
  bigger than chunk_size or dedup individually.

## Problems encountered

- **Chunk-level search let single games dominate top-k results**: since
  the vector store has no concept of "games," only independent chunk
  vectors, a game with several chunks (e.g. Hollow Knight, appearing 3x
  in one query's raw top-10) could occupy multiple slots with itself,
  crowding out genuinely different, relevant games. This was the
  single highest-impact bug found in the project - fixed by the
  deduplication decision above.

- **`chunk_size`, even changed drastically (800 -> 300), had close to no
  effect on recall**: 0.421 vs 0.413 avg recall - each config won some
  queries and lost others, netting out to noise. Went in expecting
  chunk_size to be a major lever (based on the dilution failure mode -
  see below); the data didn't support that expectation. Chunking
  parameters were not the actual bottleneck for this corpus.

- **Chunks diluting into "broad spectrum" topics.** A chunk covering
  many loosely-related sub-topics (e.g. a sandbox game whose text also
  mentions its multiplayer modes, a murder-mystery mode, "thousands of
  players online") produces an embedding that sits in a vague middle
  ground - it can rank decently against many unrelated queries simply by
  touching a bit of everything. Smaller chunks didn't clearly fix this
  in practice (see chunk_size result above) - a genuine limitation of
  chunking-level fixes, not just a parameter-tuning issue.

- **Chunks naming other, unrelated entities directly in their text.**
  E.g. Garry's Mod's chunk mentions "requires Counter-Strike: Source and
  Team Fortress 2" - pulling its embedding toward the semantic
  neighborhood of those named games, even though it isn't actually
  similar to them. Harder to fix than dilution - it's a content problem,
  not a chunk-size problem. Shrinking the chunk doesn't remove the
  mention; it can even isolate it into a purer, more "concentrated"
  chunk that matches even more strongly against the wrong queries. Left
  unresolved - out of scope relative to cost/benefit at this project's
  scale.

- **Generic/overlapping vocabulary across genres, independent of
  chunking or content noise.** E.g. "relaxing farming/life sim" retrieved
  survival games (The Forest, DayZ) ahead of the actual match (Stardew
  Valley), because both genres share surface vocabulary ("build",
  "craft", "gather resources"). Confirmed this was an **embedding model
  capacity** issue, not a chunking issue - see embeddings.md - fixed by
  switching to a larger model, not by touching chunk_size or dedup.

- **"Competitive shooter" stayed the weakest eval case across every
  config tried** (0.25-0.42 recall even with the best embedding model +
  dedup). Counter-Strike 2 never appeared in the top-10 in any run.
  Combination of the dilution and cross-mention problems above (Garry's
  Mod kept surfacing) - the remaining gap looks like a content-level
  noise problem, better suited to hybrid (keyword + semantic) search or
  source-text cleanup than to a bigger embedding model or different
  chunk_size.

- **Improving retrieval exposed gaps in `expected_games` that weren't
  visible before**: after the model upgrade, Destiny 2 (which explicitly
  describes "Competitive Multiplayer" in its own text) turned out to be
  missing from the shooter query's expected list - a real omission, not
  a judgment call. (Half-Life 2, by contrast, was deliberately excluded
  despite having a Deathmatch mode - a genuine call that its primary
  identity is single-player.) Lesson: every time retrieval genuinely
  improves, it tends to surface eval-set gaps that weren't visible
  before, because better retrieval brings borderline-legitimate
  candidates to the surface more often. Eval sets need occasional
  re-review, not one-time curation.

## Results summary (Steam games corpus, 77 games, same eval set)

| Config                                                          | n_results | avg recall |
|---------------------------------------------------------------- |-----------|------------|
| chunk_size=800, overlap=50                                      | 10        | 0.421      |
| chunk_size=300, overlap=30                                      | 10        | 0.413      |
| chunk_size=800, overlap=50 + dedup                              | 10        | 0.504      |
| chunk_size=800, overlap=50 + dedup + mpnet-base-v2 (vs MiniLM)  | 10        | 0.783      |

Biggest lever, by far: embedding model. Second: deduplication. Chunk
size: negligible effect in this project.

## Is RAG even the right tool for this domain (games)?

Worth stating plainly: a public, stable-knowledge domain like Steam game
descriptions is a **weak real-world case for RAG**. A base LLM already
"knows" popular games well from training data - RAG here only helped for
learning the pipeline mechanics, not for outperforming a plain LLM
answer. In the worst observed case (Garry's Mod outranking Counter-
Strike 2), RAG actively made the answer worse than an ungrounded LLM
would likely have given.

RAG earns its complexity when the LLM genuinely cannot know the answer:
private/internal data, data that changes after the model's training
cutoff, domains requiring source traceability, or narrow non-public
jargon/documentation. None of those apply to public Steam game
descriptions - noted here as a real conclusion of this project, not just
a caveat, and the motivation for building a second RAG on a domain that
actually meets these criteria (personal project notes - see below).