"""
build_index.py — run ONCE before the demo.
Parses the 5 PDFs, builds parent/child chunks, embeds them with BGE-M3,
and saves everything to ./index so the app starts in seconds.

Usage:
    export PAPER_RAG_DATA=/Volumes/Astro/paper_rag/data
    python build_index.py
"""
import sys
from pathlib import Path
from collections import Counter

from rag_core import (PAPERS, DATA_DIR, INDEX_DIR, EMBED_MODEL,
                      extract_pdf_pages, build_chunks, save_index)


def main():
    missing = [p["file"] for p in PAPERS if not Path(DATA_DIR, p["file"]).exists()]
    if missing:
        print(f"❌ Missing PDFs in {DATA_DIR}: {missing}")
        sys.exit(1)

    print("Phase 1: extracting PDFs ...")
    parent_store, child_chunks = {}, []
    for p in PAPERS:
        pages = extract_pdf_pages(str(Path(DATA_DIR, p["file"])))
        words = sum(len(pg["text"].split()) for pg in pages)
        print(f"  {p['doc_id']:<8} {len(pages):>2} pages {words:>6,} words [{p['access_level']}]")
        build_chunks(pages, p, parent_store, child_chunks)

    print(f"\nPhase 2: {len(parent_store)} parent / {len(child_chunks)} child chunks")
    for doc_id, n in Counter(c["doc_id"] for c in child_chunks).items():
        print(f"  {doc_id:<8} {n:>4}")

    print(f"\nPhase 3: embedding with {EMBED_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([c["text"] for c in child_chunks],
                           show_progress_bar=True, normalize_embeddings=True)

    save_index(child_chunks, parent_store, vectors)
    print(f"\n✅ Index saved to {INDEX_DIR}  (vectors: {vectors.shape})")


if __name__ == "__main__":
    main()
