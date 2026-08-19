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

4. **Chunk-level search causes single documents to dominate the top-k.**
   Since Chroma has no concept of "games", only independent chunk
   vectors, a game with many chunks (e.g. one with a long
   `about_the_game`) can occupy multiple slots in the top-k with itself,
   crowding out genuinely different, relevant games. Solved by
   deduplication (see below) - this was the single highest-impact fix
   found in this project.

**Important, and reassuring:** the LLM generation step, when properly
instructed, is robust against bad context - it correctly excluded
irrelevant retrieved games (e.g. Garry's Mod, The Elder Scrolls Online)
even when they were present in the context, rather than blindly
recommending them. Retrieval quality and generation quality are
genuinely separate problems - good generation instructions don't fix
bad retrieval, but they do limit the damage.

## Diversity-aware retrieval (deduplication by parent document)

Known pattern in mature RAG frameworks - LangChain/LlamaIndex call this
"parent document retrieval," and the more general version (balancing
relevance against diversity across *all* results, not just same-document
duplicates) is **MMR (Maximal Marginal Relevance)**. Implemented manually
here, at this scale, for full control and understanding rather than
importing the abstraction.

Mechanism: over-fetch more chunks than needed from Chroma (e.g.
`n_results * 4`), then keep only the best-scoring chunk per unique game
before truncating to the actual `n_results`. Chroma returns results
already sorted by relevance, so keeping the first occurrence of each
game name is sufficient - no manual re-sorting needed.

```python
def retrieve(query: str, model, collection, n_results: int = 5) -> list[dict]:
    query_embedding = model.encode([query]).tolist()

    query_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results * 4  # over-fetch for dedup headroom
    )

    all_chunks = [...]  # flatten documents/metadatas/distances into dicts

    seen_games = {}
    for chunk in all_chunks:
        if chunk["name"] not in seen_games:
            seen_games[chunk["name"]] = chunk  # first = best, already sorted

    return list(seen_games.values())[:n_results]
```

Why over-fetch instead of dedup-then-truncate: deduping only within the
original `n_results` window would just leave you with fewer results
whenever duplicates occur (e.g. 8 unique games instead of 10) - it
doesn't give the dedup step room to actually find 10 *distinct* good
candidates. Over-fetching first, then deduping, then truncating
preserves the intended result count.

## Evaluation set

A fixed set of (query, expected_games) cases, re-run identically after
any pipeline change, to get comparable scores over time instead of
eyeballing a handful of manual queries each time.

**Ground truth quality is the ceiling on what the evaluation can tell
you.** An early version of this set used only 1-4 "obviously correct"
games per query. This produced misleadingly high recall (0.65) - not
because retrieval was good, but because the bar for "correct" was too
narrow. Games that were genuinely valid matches (e.g. Hollow Knight for
a "soulslike" query) but weren't in the hand-written expected list would
silently punish the metric if they displaced an expected game from a
size-limited top-k, without ever being credited for being a real answer.

**Fix applied:** reviewed the *entire* corpus (not just games recalled
from memory) against each query, and expanded `expected_games` to
include every game that would reasonably count as a hit - not just the
2-3 most obvious ones. This dropped the baseline recall from 0.65 to
0.42 on the exact same retrieval code - the system didn't get worse, the
measurement got honest. Also caught a real bug: "Apex Legends" was in an
early expected list but was never actually ingested into the corpus -
no retrieval change could ever have found it.

**Metrics used:**

- **Recall@k**: of the expected games, how many appeared in the top-k?
  Doesn't penalize noise, only measures presence.
- **Reciprocal Rank**: position of the *first* expected hit (1/rank).
  Complements recall - a query can have recall=1.0 while its one
  correct answer sits buried near the bottom of the results, which
  recall alone can't distinguish from a top-1 hit.

**Runs, same eval set, same corpus (77 games), comparable:**

| Config                              | n_results | avg recall |
|--------------------------------------|-----------|------------|
| chunk_size=800, overlap=50           | 10        | 0.421      |
| chunk_size=300, overlap=30           | 10        | 0.413      |
| chunk_size=800, overlap=50 + dedup   | 10        | 0.504      |

**Conclusion:** chunk_size, even changed drastically (800 -> 300), had
no meaningful effect on recall - each config won on some queries and
lost on others, netting out to noise-level difference. Diversity-aware
deduplication was the only change that produced a clear, consistent
improvement across nearly every case, by fixing failure mode 4 above.
Failure modes 1-3 remain partially present after dedup (e.g. Counter-
Strike 2 still never appears for the "competitive shooter" query) -
chunking parameters were not the actual bottleneck for this corpus.


## Embedding model choice matters more than chunking or dedup alone

After fixing the ground-truth quality issue and applying deduplication,
the single biggest lever turned out to be the embedding model itself -
bigger than chunk_size or dedup individually.

| Config                                                    | n_results | avg recall |
|------------------------------------------------------------|-----------|------------|
| chunk_size=800, overlap=50 + dedup, MiniLM (384 dim)        | 10        | 0.504      |
| chunk_size=800, overlap=50 + dedup, mpnet-base-v2 (768 dim) | 10        | 0.783      |

Switching from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` (both free,
local, via sentence-transformers - larger model, slower to encode, same
zero-cost dev setup) pushed two of the four eval cases to a perfect
1.00 recall (relaxing/life-sim and soulslike), where MiniLM had
consistently confused adjacent genres (failure mode 3 above). This
confirms failure mode 3 was primarily a **model capacity** limitation,
not something chunk_size or dedup could fix on their own - a smaller
embedding model appears to encode genre-adjacent vocabulary (e.g.
"build", "craft", "survive") too close together regardless of how the
source text is chunked.

Dedup was still necessary to see this gain clearly - without it, a
better embedding model would still lose top-k slots to the same game's
duplicate chunks.

**Remaining gap:** the "competitive shooter" query stayed the weakest
case (0.33-0.42 across all configs) even with the better model. Counter-
Strike 2 still never appeared in the top-10, and Garry's Mod (failure
modes 1+2 combined: broad-topic dilution *and* literally naming other
shooters in its text) kept surfacing. This suggests the remaining gap is
closer to failure modes 1/2 (content-level noise) than to raw embedding
model quality - the kind of problem better addressed by hybrid
(keyword + semantic) search or content cleanup, not a bigger embedding
model.

**Practical note on evaluation set maintenance:** re-running the
evaluation after this model change also surfaced a second real
ground-truth gap - Destiny 2 (explicitly described as having
"Competitive Multiplayer" in its own text) was missing from the
shooter query's `expected_games`. Half-Life 2, by contrast, was
deliberately left out despite technically including a Deathmatch mode -
a genuine judgment call (single-player campaign as its primary identity)
rather than an oversight. Worth remembering: every time retrieval
genuinely improves, it tends to expose gaps in the expected list that
weren't visible before, simply because better retrieval surfaces
borderline-legitimate candidates more often - eval sets need occasional
re-review, not just one-time curation.

## Is RAG even the right tool for this domain (games)?

Worth stating plainly: a public, stable-knowledge domain like Steam game
descriptions is a **weak real-world case for RAG**. A base LLM already
"knows" popular games well from training data - RAG here only helps for
learning the pipeline mechanics, not for outperforming a plain LLM
answer. In the worst observed case (Garry's Mod outranking Counter-
Strike 2), RAG actively made the answer worse than an ungrounded LLM
would likely have given.

RAG earns its complexity when the LLM genuinely cannot know the answer:
private/internal data, data that changes after the model's training
cutoff, domains requiring source traceability, or narrow non-public
jargon/documentation. None of those apply to public Steam game
descriptions - noted here as a real conclusion of this project, not just
a caveat.