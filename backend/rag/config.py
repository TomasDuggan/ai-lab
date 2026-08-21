from pathlib import Path

# Project routes
RAG_ROOT = Path(__file__).resolve().parent
DATA_DIR = RAG_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
POLICIES_DIR = RAW_DIR / "northwind_analytics" / "policies"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 30

# Embedding parameters
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" # all-MiniLM-L6-v2 vs all-mpnet-base-v2
CHROMA_STEAM_GAMES_COLLECTION_NAME = "steam_games"
CHROMA_HR_POLICIES_COLLECTION_NAME = "hr_policies"

# Retreival parameters
DIVERSITY_AWARE_RETREIVAL = 4 # 1 => nothing. +1 => over-retreive chunks, then dedupe them