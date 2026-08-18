from config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from chunking import generate_chunks
from generate_embeddings import generate_embeddings
from sentence_transformers import SentenceTransformer

"""
Centralize the creation of chunks, embeddings and chroma setup
"""


def main():
    generate_chunks(CHUNK_SIZE, CHUNK_OVERLAP)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    generate_embeddings(model)

    print(f"\nDone. chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print("Run embedding_eval now if you want to compare against baseline.")

if __name__ == '__main__':
    main()