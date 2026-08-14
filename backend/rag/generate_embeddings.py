import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


CHROMA_DB_PATH = Path(__file__).parent / "data" / "chroma_db"
CHROMA_COLLECTION_NAME = "steam_games"


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


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def sanitize_metadata(metadata: dict) -> dict:
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            sanitized[key] = ", ".join(str(v) for v in value)
        else:
            sanitized[key] = value
    return sanitized


def upsert_to_chroma(collection, chunks: list[dict], embeddings: np.ndarray):
    ids = [f"{chunk['source_id']}_{chunk['chunk_index']}" for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [sanitize_metadata(chunk['metadata']) for chunk in chunks]
    embeddings_list = embeddings.tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )


def main():
    chunks_file = Path(__file__).parent / "data" / "processed" / "chunks.jsonl"

    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found")
        return

    print(f"Loading chunks from {chunks_file}")
    chunks, embeddings = generate_embeddings(load_chunks(str(chunks_file)))
    print(f"Loaded {len(chunks)} chunks with embeddings of dimension {embeddings.shape[1]}")

    collection = get_chroma_collection()
    upsert_to_chroma(collection, chunks, embeddings)
    print(f"Persisted {collection.count()} chunks to Chroma at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
