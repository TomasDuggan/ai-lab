from config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, DIVERSITY_AWARE_RETREIVAL, PROCESSED_DIR, CHROMA_STEAM_GAMES_COLLECTION_NAME
from backend.rag.chunking.chunking_steam_games import generate_chunks
from backend.rag.embedding import generate_embeddings
from sentence_transformers import SentenceTransformer

"""
Centralize the creation of chunks, embeddings and chroma setup
"""
CHUNKS_FILE = PROCESSED_DIR / "steam_games_chunks.jsonl"

def main():
    generate_chunks(CHUNK_SIZE, CHUNK_OVERLAP)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    generate_embeddings(model, CHUNKS_FILE, CHROMA_STEAM_GAMES_COLLECTION_NAME)

    print(f"\nDone (STEAM GAMES). chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}, model = {EMBEDDING_MODEL_NAME}, diversity aware retreival = {DIVERSITY_AWARE_RETREIVAL}")
    print("Run embedding_eval now if you want to compare against baseline.")

if __name__ == '__main__':
    main()