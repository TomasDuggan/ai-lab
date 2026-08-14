import chromadb
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

client_ = chromadb.PersistentClient(path=Path(__file__).parent / "data" / "chroma_db")
collection_ = client_.get_collection("steam_games")

model_ = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, model, collection, n_results: int = 5) -> list[dict]:
    query_embedding = model.encode([query]).tolist()
    
    query_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    results = []

    # [0] get only the first query response (in this specific example, I only use one query: model.encode([query]).tolist())
    for doc, meta, dist in zip(query_results["documents"][0], query_results["metadatas"][0], query_results["distances"][0]):
        res = {}

        res["name"] = meta["name"]
        res["genres"] = [genre.strip() for genre in meta["genres"].split(",")]
        res["chunk"] = doc
        res["short_description"] = meta["short_description"]
        res["distance"] = dist
        res["score"] = 1 - dist

        results.append(res)

    return results

print(json.dumps((retrieve("Horror game with monsters", model_, collection_, 2)), indent=4))
