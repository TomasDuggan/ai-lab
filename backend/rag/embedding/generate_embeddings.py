from rag.config import DATA_DIR
import json
import numpy as np
import chromadb
import shutil

"""
Common logic for reading chunks and creating the chroma folder
"""

CHROMA_DB_PATH = DATA_DIR / "chroma_db"


def load_chunks(file_path: str) -> list[dict]:
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def generate_embeddings_from_chunks(chunks: list[dict], model) -> tuple[list, np.ndarray]:
    texts = [chunk['text'] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    return chunks, embeddings


def get_chroma_collection(collection_name: str):
    if CHROMA_DB_PATH.exists():
        shutil.rmtree(CHROMA_DB_PATH)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


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


def generate_embeddings(model, chunks_file, collection_name: str):
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found")
        return

    print(f"Loading chunks from {chunks_file}")
    chunks, embeddings = generate_embeddings_from_chunks(load_chunks(str(chunks_file)), model)
    print(f"Loaded {len(chunks)} chunks with embeddings of dimension {embeddings.shape[1]}")

    collection = get_chroma_collection(collection_name)
    upsert_to_chroma(collection, chunks, embeddings)
    print(f"Persisted {collection.count()} chunks to Chroma at {CHROMA_DB_PATH}")
