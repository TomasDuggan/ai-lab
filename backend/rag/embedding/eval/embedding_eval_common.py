from pathlib import Path
from datetime import datetime
import json

"""
Compares the retrieved list (Chroma) vs. the Expected list (test data). Normalized.
This method measures the ratio of expected items that appear on retrieved list
0 => bad; 1 => good
"""
def recall_at_k(retrieved_names: list[str], expected_items: list[str]) -> float:
    retrieved_set = set(retrieved_names)
    expected_set = set(expected_items)
    hits = retrieved_set & expected_set
    return len(hits) / len(expected_set) if expected_set else 0.0

"""
This method measures how high are the expected items in the list, in relation of retrieved items
0 => bad; 1 => good
"""
def rank_noise(retrieved_names: list[str], expected_items: list[str]) -> float:
    expected_set = set(expected_items)
    for i, name in enumerate(retrieved_names):
        if name in expected_set:
            return 1 / (i + 1)
    return 0.0

def save_evaluation_run(results: list[dict], avg_recall: float, n_results: int, history_path: Path, run_label: str = ""):
    run = {
        "timestamp": datetime.now().isoformat(),
        "label": run_label,
        "n_results": n_results,
        "avg_recall": avg_recall,
        "cases": results,
    }

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")

    print(f"Saved run to {history_path}")
