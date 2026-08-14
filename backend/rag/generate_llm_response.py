import os
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import retrieve

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

"""
Takes the retreived chunks and returns a useful string for the LLM prompt.
Format (str):
    [Half-Life 2] Genres: Action. "The Seven Hour War is lost..."
    [Dead by Daylight] Genres: Horror. "Death Is Not an Escape..."
    ...
"""
def build_context(retreived_chunks: list[dict]) -> str:
    parts = []

    for chunk in retreived_chunks:
        parts.append(
            f"[{chunk['name']}]. " 
            f"Genres: {", ".join(chunk['genres'])}. "
            f"Description: {chunk['short_description']}. "
            f"Chunk data: {chunk['chunk']}."
        )

    return "\n\n".join(parts)

"""
Build msgs for OpenAI
Format:
    messages = [
        {"role": "system", "content": "Sos un asistente que recomienda videojuegos..."},
        {"role": "user", "content": f"Contexto:\n{context_text}\n\nPregunta: {user_query}"}
    ]
"""
def build_messages(user_query: str, rag_context: str) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": "You are a video game recommendation assistant. Answer the user's question using ONLY the information provided in the context below. If the context doesn't contain enough information to answer, say so explicitly instead of guessing or using outside knowledge. Keep answers concise and mention the specific game names from the context when relevant."
        },
        {
            "role": "user",
            "content": f"Context: {rag_context}\n\n. User query: {user_query}"
        }
    ]

    return messages
    
"""
Call OpenRouter with the OpenAI client. Returns response.choices[0].message.content
"""
def generate_answer(messages: list[dict]) -> str:
    print("Taking it to the LLM...")

    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY")
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages
    )

    print("LLM has responded")

    return response.choices[0].message.content

def main():
    client_ = chromadb.PersistentClient(path=Path(__file__).parent / "data" / "chroma_db")
    collection = client_.get_collection("steam_games")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    user_query = "open world game with dragons and magic"
    chunks = retrieve(user_query, model, collection, 3)

    for i, chunk in enumerate(chunks):
        similarity = f"{chunk['score']:.3}"
        print(f"{i}: {chunk['name']}, score: {similarity}")

    context: str = build_context(chunks)
    msgs: list[dict] = build_messages(user_query, context)
    llm_answer = generate_answer(msgs)
    print(llm_answer)

if __name__ == "__main__":
    main()