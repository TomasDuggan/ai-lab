"""
Generate chunks for HR policy markdown files.
"""
from rag.config import POLICIES_DIR, PROCESSED_DIR
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter

OUTPUT_FILE = PROCESSED_DIR / 'hr_policies_chunks.jsonl'
HEADERS_TO_SPLIT_ON = [("#", "policy_name"), ("##", "section")]


def generate_chunks():
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

    stats = {
        "documents_processed": 0,
        "total_chunks": 0,
        "documents_with_multiple_chunks": 0
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for md_file in sorted(POLICIES_DIR.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            stats["documents_processed"] += 1

            docs = splitter.split_text(text)

            if len(docs) > 1:
                stats["documents_with_multiple_chunks"] += 1

            for chunk_index, doc in enumerate(docs):
                chunk_obj = {
                    "text": doc.page_content,
                    "source_id": md_file.stem,
                    "chunk_index": chunk_index,
                    "metadata": {
                        "policy_name": doc.metadata.get("policy_name", md_file.stem),
                        "section": doc.metadata.get("section")
                    }
                }
                outfile.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                stats["total_chunks"] += 1

    avg_chunks = stats["total_chunks"] / stats["documents_processed"] if stats["documents_processed"] > 0 else 0

    print("\n=== Chunking Summary ===")
    print(f"Total de políticas procesadas: {stats['documents_processed']}")
    print(f"Total de chunks generados: {stats['total_chunks']}")
    print(f"Promedio de chunks por política: {avg_chunks:.2f}")
    print(f"Políticas con más de 1 chunk: {stats['documents_with_multiple_chunks']}")
    print(f"Chunks guardados en: {OUTPUT_FILE.resolve()}")
