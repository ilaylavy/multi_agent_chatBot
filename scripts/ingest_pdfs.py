"""
scripts/ingest_pdfs.py — Ingest all PDFs into ChromaDB.

Scans two directories in order:
  1. tests/fixtures/pdfs/  — committed test fixture PDFs (always present in git)
  2. data/pdfs/            — real user-supplied PDFs (gitignored, optional)

For each PDF found (deduped by filename, fixtures take precedence):
  1. Parse with pymupdf4llm (handles tables and complex layouts)
  2. Chunk text into ~500-char segments with 50-char overlap
  3. Upsert chunks into a ChromaDB collection named after the PDF stem
     (e.g. travel_policy_2024.pdf → collection "travel_policy_2024")
  4. Store metadata: source_pdf, page_number, chunk_index

Collection name matches the source_id used in the manifest, so the
Librarian's retriever.search(source_id=...) will find the right collection.

Run with: python -m scripts.ingest_pdfs
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pymupdf4llm

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
FIXTURE_PDFS_DIR  = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
PDFS_DIR          = PROJECT_ROOT / "data" / "pdfs"
CHROMA_DIR        = PROJECT_ROOT / "data" / "chroma_db"

CHUNK_SIZE    = 500   # target characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks of ~size characters.
    Splits prefer whitespace boundaries near the target size.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Walk back to the nearest whitespace to avoid mid-word cuts
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # step back by overlap for the next chunk
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, client: chromadb.PersistentClient) -> int:
    """
    Parse a single PDF and upsert all chunks into ChromaDB.
    Returns the total number of chunks ingested.
    """
    collection_name = pdf_path.stem   # e.g. "travel_policy_2024"

    # get_or_create so re-running is safe
    collection = client.get_or_create_collection(name=collection_name)

    # pymupdf4llm returns a list of dicts, one per page
    pages: list[dict] = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    ids:        list[str]  = []
    texts:      list[str]  = []
    metadatas:  list[dict] = []

    global_chunk_index = 0   # unique across the whole document

    for page in pages:
        page_number  = page.get("metadata", {}).get("page", 0) + 1   # 1-indexed
        page_text    = page.get("text", "")
        page_chunks  = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_index, chunk in enumerate(page_chunks):
            chunk_id = f"{collection_name}_{global_chunk_index}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "source_pdf":  pdf_path.name,
                "page_number": page_number,
                "chunk_index": chunk_index,
            })
            global_chunk_index += 1

    if ids:
        # Upsert in batches of 100 (ChromaDB recommendation for large ingests)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    return len(ids)


def run_ingestion() -> None:
    # Collect PDFs from both directories; fixtures take precedence (inserted first).
    # A dict keyed by stem deduplicates: if the same filename exists in both dirs,
    # the fixture version wins.
    seen: dict[str, Path] = {}
    for source_dir in (FIXTURE_PDFS_DIR, PDFS_DIR):
        if source_dir.exists():
            for p in sorted(source_dir.glob("*.pdf")):
                seen.setdefault(p.stem, p)   # fixture inserted first, real-data skipped if duplicate

    pdf_files = list(seen.values())
    if not pdf_files:
        print(f"No PDF files found in {FIXTURE_PDFS_DIR} or {PDFS_DIR}")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    print(f"ChromaDB path : {CHROMA_DIR}")
    print(f"PDFs found    : {len(pdf_files)}  "
          f"(fixtures: {FIXTURE_PDFS_DIR}, real data: {PDFS_DIR})\n")

    total_chunks = 0
    for pdf_path in pdf_files:
        n = ingest_pdf(pdf_path, client)
        total_chunks += n
        print(f"  {pdf_path.name}")
        print(f"    Collection : {pdf_path.stem}")
        print(f"    Chunks     : {n}")

        # Verify — query the collection and confirm count
        collection = client.get_collection(name=pdf_path.stem)
        stored = collection.count()
        status  = "OK" if stored == n else f"WARNING: stored={stored}, expected={n}"
        print(f"    Verified   : {stored} chunks in ChromaDB  [{status}]")
        print()

    print(f"Total chunks ingested: {total_chunks}")

    # Spot-check: confirm the key sentence is retrievable
    print("\nSpot-check — querying 'travel_policy_2024' for Business Class entitlements...")
    collection = client.get_collection(name="travel_policy_2024")
    results = collection.query(
        query_texts=["Business Class entitlement clearance level A"],
        n_results=3,
    )
    for doc in results["documents"][0]:
        print(f"  [{doc[:120]}...]" if len(doc) > 120 else f"  [{doc}]")


if __name__ == "__main__":
    run_ingestion()
