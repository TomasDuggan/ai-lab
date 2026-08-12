# Personal notes/theory

---

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