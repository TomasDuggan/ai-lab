from rag.config import EMBEDDING_MODEL_NAME, CHROMA_HR_POLICIES_COLLECTION_NAME, DIVERSITY_AWARE_RETREIVAL
from backend.rag.retrieve.retrieve_hr_policies import retrieve
from backend.rag.generate_hr_policies_rag_data import CHUNK_SIZE, CHUNK_OVERLAP
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
        retrieved_policies = retrieve(case["query"], model, collection, n_results)
        retrieved_names = [policy["policy_name"].strip() for policy in retrieved_policies]
        recall = recall_at_k(retrieved_names, case["expected_policies"])
        rn = rank_noise(retrieved_names, case["expected_policies"])

        results.append({
            "query": case["query"],
            "recall": round(recall, 3),
            "rank_noise": round(rn, 3),
            "retrieved": retrieved_names,
            "expected": case["expected_policies"]
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
            "query": "How much paid time off do I accrue and how many days can I carry over to next year?",
            "expected_policies": ["PTO / Vacation Policy"],
        },
        {
            "query": "What is the process and notice period for resigning from my job?",
            "expected_policies": ["Termination & Resignation Policy"],
        },
        {
            "query": "Can I work remotely and what equipment will the company provide?",
            "expected_policies": ["Remote Work Policy"],
        },
    ]

    client = chromadb.PersistentClient(path=DATA_DIR / "chroma_db")
    collection = client.get_collection(CHROMA_HR_POLICIES_COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    n_results = 10
    run_label = f"Chunk size = {CHUNK_SIZE}, overlap = {CHUNK_OVERLAP}. Model = {EMBEDDING_MODEL_NAME}. Diversity aware retreival = {DIVERSITY_AWARE_RETREIVAL}"

    results = run_evaluation(eval_set, model, collection, n_results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    save_evaluation_run(results, avg_recall, n_results, PROCESSED_DIR / "eval_history_hr_policies.jsonl", run_label)

if __name__ == "__main__":
    main()
