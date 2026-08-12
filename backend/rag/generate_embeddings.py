import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


"""Load chunks from JSONL file."""
def load_chunks(file_path: str) -> list[dict]:
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


"""Generate embeddings for all chunks."""
"""
Notes:
embeddings => Matrix(N, X), where N is the amount of chunks and X depends on the model (all-MiniLM-L6-v2 is 384).
It basically results in a matrix, each row is a chunk, each column holds a normalized number, a whole row represents the "semantic" of a chunk.
"""
def generate_embeddings(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2") -> tuple[list, np.ndarray]:
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk['text'] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    return chunks, embeddings


"""Find n most similar and n most opposite chunks to reference chunk."""
"""
Notes:
- N => The amount of elements (columns) of each chunk, 384 for all-MiniLM-L6-v2
- embeddings[0] => 1D Array. An array of the N numbers that represent the semantic of a chunk.
- .reshape(1, -1) => Matrix of 1 row and N columns (cosine_similarity expects a matrix). The -1 arg is so numpy calculates the amount of columns.
- cosine_similarity => in this case, returns (1, amount_of_chunks), a 1D array where each position is the chunk similarity.
    Ej: [[0.999, 0.73, 0.12, 0.91, ...]]. 0.999 => chunk 0, 0.73 => chunk 1.
    [0] so we have a 1D array.
    So, now: similarities[i] relates to chunks[i]. similarities[7] => similarity between ref_chunk and chunks[7]
- argsort sorts (asc) but returns indexes.
"""
def find_similar_and_opposite(embeddings: np.ndarray, ref_idx: int = 0, n: int = 3) -> tuple[list, list]:
    if (n <= 0):
        return ([], [])

    ref_embedding = embeddings[ref_idx].reshape(1, -1)
    similarities = cosine_similarity(ref_embedding, embeddings)[0]

    # Exclude reference chunk itself
    similarities[ref_idx] = -np.inf

    # Get indices of most similar and most opposite
    similar_indices = np.argsort(similarities)[-n:][::-1]  # Top n, descending
    opposite_indices = np.argsort(similarities)[1:n+1]  # Bottom n, excluding first (-inf)

    return similar_indices.tolist(), opposite_indices.tolist()


"""Print comparison of similar and opposit~nks."""
def print_comparison(chunks: list[dict], embeddings: np.ndarray, ref_idx: int = 0):
    similar_idx, opposite_idx = find_similar_and_opposite(embeddings, ref_idx)

    ref_chunk = chunks[ref_idx]
    ref_embedding = embeddings[ref_idx].reshape(1, -1)

    print("\n" + "="*80)
    print(f"REFERENCE CHUNK (index {ref_idx})")
    print("="*80)
    print(f"Game: {ref_chunk['metadata'].get('name', 'Unknown')}")
    print(f"Text: {ref_chunk['text'][:150]}...")

    print("\n" + "="*80)
    print("3 MOST SIMILAR CHUNKS")
    print("="*80)
    for i, idx in enumerate(similar_idx, 1):
        similarity = cosine_similarity(ref_embedding, embeddings[idx].reshape(1, -1))[0][0]
        chunk = chunks[idx]
        print(f"\n{i}. Similarity: {similarity:.4f}")
        print(f"   Game: {chunk['metadata'].get('name', 'Unknown')}")
        print(f"   Text: {chunk['text'][:100]}...")

    print("\n" + "="*80)
    print("3 MOST OPPOSITE CHUNKS")
    print("="*80)
    for i, idx in enumerate(opposite_idx, 1):
        similarity = cosine_similarity(ref_embedding, embeddings[idx].reshape(1, -1))[0][0]
        chunk = chunks[idx]
        print(f"\n{i}. Similarity: {similarity:.4f}")
        print(f"   Game: {chunk['metadata'].get('name', 'Unknown')}")
        print(f"   Text: {chunk['text'][:100]}...")


def main():
    chunks_file = Path(__file__).parent / "data" / "processed" / "chunks.jsonl"

    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found")
        return

    print(f"Loading chunks from {chunks_file}")
    chunks, embeddings = generate_embeddings(load_chunks(str(chunks_file)))
    print(f"Loaded {len(chunks)} chunks with embeddings of dimension {embeddings.shape[1]}")

    # Test with first chunk as reference
    print_comparison(chunks, embeddings, ref_idx=0)

    print("\n" + "="*80)
    print("EMBEDDING STATISTICS")
    print("="*80)
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
