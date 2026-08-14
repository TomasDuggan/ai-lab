from chunking import generate_chunks
from generate_embeddings import generate_embeddings
from sentence_transformers import SentenceTransformer

"""
Centralize the creation of chunks, embeddings and chroma setup
"""

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50


def main():
    generate_chunks(CHUNK_SIZE, CHUNK_OVERLAP)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    generate_embeddings(model)

    print(f"\nDone. chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print("Run evaluation now to compare against baseline.")

if __name__ == '__main__':
    main()