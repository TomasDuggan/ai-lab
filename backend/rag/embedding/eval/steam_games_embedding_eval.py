from rag.config import EMBEDDING_MODEL_NAME, CHROMA_STEAM_GAMES_COLLECTION_NAME, DIVERSITY_AWARE_RETREIVAL
from backend.rag.retrieve.retrieve_steam_games import retrieve
from backend.rag.generate_steam_games_rag_data import CHUNK_SIZE, CHUNK_OVERLAP
from backend.rag.embedding.eval.embedding_eval_common import recall_at_k, rank_noise, save_evaluation_run

from config import PROCESSED_DIR, DATA_DIR

import chromadb
from sentence_transformers import SentenceTransformer

"""
Evaluation set for the embeddings
Each time I change the embeddings db (different chunk size for example) this script runs,
logging measurements and saving data in a json for later inspection.
"""

def run_evaluation(eval_set: list[dict], model, collection, n_results: int) -> list[dict]:
    results = []

    for case in eval_set:
        retrieved_games = retrieve(case["query"], model, collection, n_results)
        retrieved_names = [game["name"].strip() for game in retrieved_games]
        recall = recall_at_k(retrieved_names, case["expected_games"])
        rn = rank_noise(retrieved_names, case["expected_games"])

        results.append({
            "query": case["query"],
            "recall": round(recall, 3),
            "rank_noise": round(rn, 3),
            "retrieved": retrieved_names,
            "expected": case["expected_games"]
        })

    print()
    for r in results:
        print(f"[{r['recall']:.2f}] {r['query']}")
        print(f"Retrieved -> {r['retrieved']}")
        print(f"Expected -> {r['expected']}\n")
    
    return results


def main():
    eval_set = [
        {
            "query": "I want a fast-paced multiplayer shooter, something competitive",
            "expected_games": [
                "Counter-Strike 2", "Counter-Strike: Source", "Apex Legends", "Farlight 84",
                "Titanfall® 2", "Team Fortress 2", "Battlefield™ V", "Battlefield™ 2042",
                "Call of Duty®", "HELLDIVERS™ 2", "ULTRAKILL", "NARAKA: BLADEPOINT", "Destiny 2"
            ],
            # Destiny 2 is PvE-leaning but has real competitive PvP - borderline include.
        },
        {
            "query": "relaxing farming and life simulation game",
            "expected_games": ["Stardew Valley", "DAVE THE DIVER"],
            # Dave the Diver mixes sim/management with combat, still closer to "relaxing sim" than most.
        },
        {
            "query": "dark souls-like game with challenging combat and bosses",
            "expected_games": [
                "DARK SOULS™: REMASTERED", "Sekiro™: Shadows Die Twice - GOTY Edition",
                "ELDEN RING", "Hollow Knight", "Dead Cells", "Hades",
            ],
            # These share punishing combat, boss-focused design, and death-as-progression - the core soulslike DNA.
        },
        {
            "query": "open world game with dragons and magic",
            "expected_games": [
                "The Elder Scrolls V: Skyrim", "The Elder Scrolls V: Skyrim Special Edition",
                "Dragon's Dogma 2", "Baldur's Gate 3", "The Elder Scrolls® Online",
            ],
            # BG3 is party-based/D&D not fully "open world" in the sandbox sense, but has
            # dragons/magic centrally and large explorable areas - kept as a judgment call.
        },
    ]

    client = chromadb.PersistentClient(path=DATA_DIR / "chroma_db")
    collection = client.get_collection(CHROMA_STEAM_GAMES_COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    n_results = 10
    run_label = f"Chunk size = {CHUNK_SIZE}, overlap = {CHUNK_OVERLAP}. Model = {EMBEDDING_MODEL_NAME}. Diversity aware retreival = {DIVERSITY_AWARE_RETREIVAL}"

    results = run_evaluation(eval_set, model, collection, n_results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    save_evaluation_run(results, avg_recall, n_results, PROCESSED_DIR / "eval_history_steam_games.jsonl", run_label)

if __name__ == "__main__":
    main()
