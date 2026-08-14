from retrieve import retrieve

from pathlib import Path
import json
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

"""
Evaluation set for the embeddings
Each time I change the embeddings db (different chunk size for example) this script runs, 
logging measurements and saving data in a json for later inspection.
"""

"""
Compares the retreived list (Chroma) vs. the Expected list (test data). Normalized.
This method measures the ratio of expected games that appear on retreived_games
0 => bad; 1 => good
"""
def recall_at_k(retrieved_names: list[str], expected_games: list[str]) -> float:
    retrieved_set = set(retrieved_names)
    expected_set = set(expected_games)
    hits = retrieved_set & expected_set
    return len(hits) / len(expected_set) if expected_set else 0.0

"""
This method measures how high are the expected games in the list, in relation of retrieved games
0 => bad; 1 => good
"""
def rank_noise(retrieved_names: list[str], expected_games: list[str]) -> float:
    expected_set = set(expected_games)
    for i, name in enumerate(retrieved_names):
        if name in expected_set:
            return 1 / (i + 1)
    return 0.0

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

def save_evaluation_run(results: list[dict], avg_recall: float, n_results: int, run_label: str = ""):
    run = {
        "timestamp": datetime.now().isoformat(),
        "label": run_label,          # ej "baseline" o "chunk_size=800->500"
        "n_results": n_results,
        "avg_recall": avg_recall,
        "cases": results,
    }

    history_path = Path(__file__).parent / "data" / "processed" / "eval_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")

    print(f"Saved run to {history_path}")

def main():
    eval_set = [
        {
            "query": "I want a fast-paced multiplayer shooter, something competitive",
            "expected_games": ["Counter-Strike 2", "Apex Legends", "Farlight 84", "Titanfall® 2"],
        },
        {
            "query": "relaxing farming and life simulation game",
            "expected_games": ["Stardew Valley"],
        },
        {
            "query": "dark souls-like game with challenging combat and bosses",
            "expected_games": ["DARK SOULS™: REMASTERED", "Sekiro™: Shadows Die Twice - GOTY Edition", "ELDEN RING"],
        },
        {
            "query": "open world game with dragons and magic",
            "expected_games": ["The Elder Scrolls V: Skyrim", "Dragon's Dogma 2", "Baldur's Gate 3"],
        },
    ]

    client = chromadb.PersistentClient(path=Path(__file__).parent / "data" / "chroma_db")
    collection = client.get_collection("steam_games")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    n_results = 15
    run_label = "baseline"

    results = run_evaluation(eval_set, model, collection, n_results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    save_evaluation_run(results, avg_recall, n_results, run_label)

if __name__ == "__main__":
    main()
