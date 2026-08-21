from config import DIVERSITY_AWARE_RETREIVAL

"""
Retrieve chunks from de VectorDB based on closeness
"""
def retrieve(user_query: str, model, collection, n_results: int = 5) -> list[dict]:
    query_embedding = model.encode([user_query]).tolist()
    
    query_results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results * DIVERSITY_AWARE_RETREIVAL # Go overboard for diversity
    )

    chunks = []

    # [0] get only the first query response (in this specific example, I only use one query: model.encode([user_query]).tolist())
    for doc, meta, dist in zip(query_results["documents"][0], query_results["metadatas"][0], query_results["distances"][0]):
        chunks.append({
            "policy_name": meta["policy_name"],
            "section": meta["section"],
            "chunk": doc,
            "distance": dist,
            "score": 1 - dist
        })


    deduped_chunks = {}
    for chunk in chunks:
        name = chunk["policy_name"]
        if name not in deduped_chunks:
            deduped_chunks[name] = chunk

    deduped_list = list(deduped_chunks.values())[:n_results]

    print(f"Found {len(deduped_list)} chunks for query {user_query}")

    return deduped_list
