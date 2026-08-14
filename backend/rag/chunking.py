"""
Generate chunks for games.json
"""

import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = Path(__file__).parent / 'data' / 'raw' / 'games.jsonl'
OUTPUT_FILE = Path(__file__).parent / 'data' / 'processed' / 'chunks.jsonl'

# Patrones a no romper (abreviaciones comunes)
ABBREVIATIONS = ["vs", "sr", "sra", "dr", "dra", "etc", "aprox", "pag", "pp"]
PLACEHOLDER_TEMPLATE = "§ABBREV_{}§"

def protect_abbreviations(text):
    """Reemplaza abreviaciones con marcadores temporales para evitar splits falsas."""
    protected = text
    replacements = {}
    for i, abbrev in enumerate(ABBREVIATIONS):
        pattern = rf'\b{abbrev}\.'
        placeholder = PLACEHOLDER_TEMPLATE.format(i)
        if re.search(pattern, protected, re.IGNORECASE):
            protected = re.sub(pattern, placeholder, protected, flags=re.IGNORECASE)
            replacements[i] = abbrev + "."
    return protected, replacements

def restore_abbreviations(text, replacements):
    """Restaura las abreviaciones originales."""
    restored = text
    for i, abbrev in replacements.items():
        placeholder = PLACEHOLDER_TEMPLATE.format(i)
        restored = restored.replace(placeholder, abbrev)
    return restored

def clean_chunk_boundaries(chunks):
    """Mueve puntuación del inicio de chunks siguientes al final del anterior."""
    if not chunks:
        return chunks

    cleaned = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            # Si este chunk empieza con puntuación, moverla al anterior
            match = re.match(r'^([.!?,;:-]+)\s*', chunk)
            if match:
                punct = match.group(1)
                cleaned[-1] = cleaned[-1] + punct
                chunk = chunk[len(match.group(0)):]

        cleaned.append(chunk.strip())

    return cleaned

def generate_chunks(chunk_size: int = 800, chunk_overlap: int = 50):
    # Inicializar splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    stats = {
        "games_processed": 0,
        "total_chunks": 0,
        "games_with_multiple_chunks": 0
    }

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
        open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            game = json.loads(line)
            stats["games_processed"] += 1

            # Extraer campos
            about_text = game.get("about_the_game", "")
            if not about_text:
                continue

            # Campos a mantener como metadata
            metadata = {
                "name": game.get("name"),
                "genres": game.get("genres", []),
                "developers": game.get("developers", []),
                "release_date": game.get("release_date"),
                "short_description": game.get("short_description")
            }

            # Splitear el texto
            protected_text, replacements = protect_abbreviations(about_text)
            chunks_text = splitter.split_text(protected_text)
            chunks_text = [restore_abbreviations(chunk, replacements) for chunk in chunks_text]
            chunks_text = clean_chunk_boundaries(chunks_text)

            if len(chunks_text) > 1:
                stats["games_with_multiple_chunks"] += 1

            # Generar objetos de chunk
            for chunk_index, text in enumerate(chunks_text):
                chunk_obj = {
                    "text": text,
                    "source_id": str(game.get("appid")),
                    "chunk_index": chunk_index,
                    "metadata": metadata
                }
                outfile.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                stats["total_chunks"] += 1

    # Resumen
    avg_chunks = stats["total_chunks"] / stats["games_processed"] if stats["games_processed"] > 0 else 0

    print("\n=== Chunking Summary ===")
    print(f"Total de juegos procesados: {stats['games_processed']}")
    print(f"Total de chunks generados: {stats['total_chunks']}")
    print(f"Promedio de chunks por juego: {avg_chunks:.2f}")
    print(f"Juegos con más de 1 chunk: {stats['games_with_multiple_chunks']}")
    print(f"Chunks guardados en: {OUTPUT_FILE.resolve()}")
