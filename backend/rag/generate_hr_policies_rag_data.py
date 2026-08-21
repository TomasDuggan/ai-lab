from rag.config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, DIVERSITY_AWARE_RETREIVAL, PROCESSED_DIR, CHROMA_HR_POLICIES_COLLECTION_NAME
from rag.chunking.chunking_hr_policies import generate_chunks
from rag.embedding.generate_embeddings import generate_embeddings
from sentence_transformers import SentenceTransformer

"""
Centralize the creation of chunks, embeddings and chroma setup.
Run => parado en backend: python -m rag.generate_hr_policies_rag_data
"""

CHUNKS_FILE = PROCESSED_DIR / "hr_policies_chunks.jsonl"

def main():
    generate_chunks()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    generate_embeddings(model, CHUNKS_FILE, CHROMA_HR_POLICIES_COLLECTION_NAME)

    print(f"\nDone (HR POLICIES). chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}, model = {EMBEDDING_MODEL_NAME}, diversity aware retreival = {DIVERSITY_AWARE_RETREIVAL}")
    print("Run embedding_eval now if you want to compare against baseline.")

if __name__ == '__main__':
    main()