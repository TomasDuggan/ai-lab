# Chunking

## What it is

A chunk is a fragment of text that becomes an independent unit in the vector 
store: it gets its own embedding and can be retrieved on its own.
Question it answers: what's the minimum unit of information that makes sense 
to retrieve by itself?

## Why it matters

- LLM context limit: you can't send the whole corpus, only what's relevant.
- Semantic density: a chunk mixing topics produces an averaged, ambiguous 
  embedding → worse retrieval. A focused chunk = sharper vector.
- Size trade-off: too small loses context; too large dilutes the embedding 
  and adds noise to what gets sent to the LLM.
- Overlap: repeating the tail of one chunk at the start of the next avoids 
  losing ideas cut right at the boundary. Costs redundancy and more vectors.

## Strategies (simple to complex)

- Fixed-size: cuts every N characters/tokens, with or without overlap. 
  Simple but ignores text structure.
- Recursive character splitting: tries to cut respecting a hierarchy of 
  separators (\n\n → \n → ". " → " "). De facto default (LangChain's 
  RecursiveCharacterTextSplitter). It's a hybrid: has a fixed cap 
  (chunk_size) but looks for the nearest natural break instead of forcing 
  a cut.
- Semantic chunking: uses embeddings to detect topic shifts between 
  sentences and cuts there. Better quality on text that mixes topics with 
  no clear markup, but more expensive. Overkill for short documents 
  (~200 words).
- Document-as-chunk: if the document is already short, don't split it — 
  it's one whole chunk.

## Libraries

- langchain.text_splitter (RecursiveCharacterTextSplitter, 
  CharacterTextSplitter, TokenTextSplitter) — de facto standard.
- tiktoken — count real tokens instead of characters.
- llama-index — has its own NodeParser, similar options.

## Decide per field, not per dataset

You don't pick "one strategy" globally — you decide per field, based on 
its role:
- Atomic fields (name, appid): never chunked, go in whole.
- Long/variable content field (about_the_game): the real candidate for 
  splitting.
- Structured fields (genres, developers, release_date): not chunked or 
  embedded, go in as metadata for filtering.

## Indexing vs metadata

- Indexing = generate an embedding of the text → becomes searchable by 
  meaning.
- Metadata = text/values that travel with the chunk but aren't vectorized 
  → used for filtering (where genre == "Action") or enriching the response.
- A field should be indexed if it adds unique searchable meaning. If it 
  overlaps heavily with another already-indexed field (e.g. 
  short_description vs about_the_game), indexing both just produces 
  duplicate results without adding coverage — better to keep it as 
  metadata/preview only.

## Base snippet

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks_text = splitter.split_text(game["about_the_game"]) # only "about_the_game" gets chunked
for chunk_index, text in enumerate(chunks_text):
    chunk_obj = {
        "text": text, # the chunk data
        "source_id": str(game.get("appid")), # to know where does a group of chunks come from
        "chunk_index": chunk_index, # the index of the chunk in its group
        "metadata": metadata # extra, for classic filtering
    }
    outfile.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
```

## Decisions made in this project

- **chunk_size=800, overlap=50 as the working baseline (not chunk_size=300)**:
  tested both against the same evaluation set, on the same corpus. Recall
  was nearly identical (0.421 vs 0.413) - each config won some queries and
  lost others, netting out to noise-level difference. Kept 800/50 as the
  baseline since smaller chunks produced no measurable benefit and chunk
  count per game was already high (avg ~5 chunks for ~150-250 word texts)
  at smaller sizes.

- **Only `about_the_game` gets chunked; `short_description` is metadata
  only, never indexed**: both fields describe the same game and overlap
  heavily in content. Indexing both would produce near-duplicate results
  from the same source without adding retrieval coverage - so
  `short_description` is kept as metadata (for previews/display) rather
  than embedded.

- **Custom abbreviation protection before splitting**: added a
  pre/post-processing step (`protect_abbreviations` /
  `restore_abbreviations`) that temporarily replaces known abbreviations
  (e.g. "vs.") with placeholders before running the splitter, so the
  ". " separator doesn't treat them as sentence boundaries. Motivated by
  an observed real split: "Combine vs. Resistance teamplay" got cut into
  "...try Combine vs" / ". Resistance teamplay" at chunk_size=500.

## Problems encountered

- **Chunk-boundary cleanup silently canceled `chunk_overlap`**: added
  `clean_chunk_boundaries()` to fix a cosmetic issue - chunks sometimes
  started with a stray punctuation mark (e.g. ". And a lot of people..."),
  left over from where the splitter cut. The fix works by moving leading
  punctuation from a chunk to the end of the previous one, trimming it
  off the current chunk's start. Side effect not anticipated: LangChain's
  overlap almost always starts by repeating text right after the cut
  point - which frequently begins with that same punctuation. So the
  "cleanup" was, most of the time, also deleting the intended overlap.
  Confirmed by checking consecutive chunks from the same game and finding
  no repeated tail/head text between them, despite `chunk_overlap` being
  set. Net effect: prettier chunk boundaries, but overlap wasn't actually
  doing anything in practice. Left unresolved / accepted for this
  project's scale - noted as a real trade-off rather than fixed, since
  documents are short enough that the lack of overlap likely has limited
  impact.