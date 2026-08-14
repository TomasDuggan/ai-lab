"""
Retrieve chunks drom de VectorDB based on closeness
"""


def retrieve(user_query: str, model, collection, n_results: int = 5) -> list[dict]:
    print("Retrieving relevant chunks...")

    query_embedding = model.encode([user_query]).tolist()
    
    query_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    results = []

    # [0] get only the first query response (in this specific example, I only use one query: model.encode([user_query]).tolist())
    for doc, meta, dist in zip(query_results["documents"][0], query_results["metadatas"][0], query_results["distances"][0]):
        results.append({
            "name": meta["name"],
            "genres": [genre.strip() for genre in meta["genres"].split(",")],
            "chunk": doc,
            "short_description": meta["short_description"],
            "distance": dist,
            "score": 1 - dist
        })
        
    print(f"Found {len(results)} chunks.")

    return results
