# Embeddings

## What it is

An embedding is a numerical representation of a piece of text as a vector.
Its goal is to place semantically similar texts close to each other in a
high-dimensional vector space.

For example, `all-MiniLM-L6-v2` produces a 384-dimensional embedding.

A collection of embeddings is stored as a 2D NumPy array (a matrix):

- Rows = chunks.
- Columns = dimensions of the embedding.
- Each row = one complete embedding/vector, the "semantic" of a single chunk represented as an array of numbers.
- Each column = one numerical dimension of that vector.
- Individual dimensions usually have no useful human-interpretable meaning;
  the semantic information is distributed across the whole vector.

If there are N chunks and the model produces X-dimensional embeddings:

    embeddings.shape == (N, X)

For example:

    100 chunks × 384 dimensions → (100, 384)

Each row can be thought of as a point in an X-dimensional vector space.
Although we cannot visualize 384 dimensions directly, we can imagine a
2D/3D space where each text becomes a point and similar texts tend to be
closer together.

## Why it matters

- Embeddings turn text into something that can be compared mathematically.
- Semantic similarity can be measured even when texts use different words.
  "first-person shooter" and "FPS game" can end up close in vector space.
- This enables semantic retrieval: embed the user's query, compare it against
  the stored chunk embeddings, and retrieve the most similar chunks.
- The embedding represents the semantics of the whole text, so chunk quality
  directly affects embedding quality and therefore retrieval quality.
- The exact meaning of an individual number/dimension is generally not useful
  to interpret. The meaning is distributed across the complete vector.

## Embedding shape

Given:

    embeddings.shape == (N, X)

`embeddings[i]` selects one row/chunk and returns a 1D NumPy array:

    embeddings[i].shape == (X,)

For example:

    embeddings.shape       → (100, 384)
    embeddings[0].shape   → (384,)

The embedding itself is therefore a 1D vector containing 384 numbers.

Sometimes a function expects a collection of vectors rather than one vector.
`reshape(1, -1)` converts:

    (384,) → (1, 384)

This does not change the data or the embedding. It changes its shape so that
it is treated as a matrix containing one row/vector.

`-1` means NumPy calculates that dimension automatically.

## Similarity

To find relevant chunks, the query is also converted into an embedding.
The query embedding can then be compared against all stored embeddings.

Cosine similarity measures how aligned two vectors are:

    cosine similarity ≈ 1   → very similar direction
    cosine similarity ≈ 0   → very different directions
    cosine similarity ≈ -1  → opposite directions

Geometrically, cosine similarity is based on the angle between two vectors.
It focuses on the direction of the vectors rather than simply their raw
distance.

Conceptually:

    query
      ↓
    [384 numbers]
      ↓
    compare against every chunk embedding
      ↓
    [similarity_0, similarity_1, similarity_2, ...]
      ↓
    highest scores → most similar/relevant chunks

The individual similarity values are not "percent semantic similarity".
They are scores indicating how aligned the two embedding vectors are.

## Multiple chunks

If one element produces multiple chunks:

    Element
     ├── chunk 0 → embedding 0
     └── chunk 1 → embedding 1

Each chunk gets its own embedding and becomes its own point in the vector
space.

Chunks from the same document may be close to each other if their content is
semantically related, but this is not guaranteed. The embedding model sees
the text content, not the `source_id`.

Therefore:

    source_id = 220

does not make two chunks close together. Their semantic content determines
their position in the embedding space.

## Generating embeddings

```python
def generate_embeddings(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2"
) -> tuple[list, np.ndarray]:

    model = SentenceTransformer(model_name)

    # Only the text gets embedded.
    texts = [chunk["text"] for chunk in chunks]

    # One embedding/vector is generated for each text.
    embeddings = model.encode(texts)

    return chunks, embeddings
```