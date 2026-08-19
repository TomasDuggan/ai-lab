# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 30

# Embedding parameters
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2" # all-MiniLM-L6-v2 vs all-mpnet-base-v2
CHROMA_COLLECTION_NAME = "steam_games"

# Retreival parameters
DIVERSITY_AWARE_RETREIVAL = 4 # 1 => nothing, +1 => retreive more chunks, then dedupe them