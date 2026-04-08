"""
scripts/ingest_pdfs.py — Ingest all PDFs into ChromaDB and both manifests.

Scans two directories in order:
  1. tests/fixtures/pdfs/  — committed test fixture PDFs (always present in git)
  2. data/pdfs/            — real user-supplied PDFs (gitignored, optional)

For each PDF found (deduped by filename stem, fixtures take precedence):
  Calls ingest_pdf() from ingestion/pdf_ingestor.py, which:
    - Parses the PDF with pymupdf4llm
    - Catalogs it via LLM (summary, sections, tags)
    - Writes both manifest files
    - Upserts chunks into ChromaDB (collection name == source_id == PDF stem)

Run with: python -m scripts.ingest_pdfs
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ingestion.pdf_ingestor import ingest_pdf

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
FIXTURE_PDFS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
PDFS_DIR         = PROJECT_ROOT / "data" / "pdfs"


def run_ingestion() -> None:
    # Collect PDFs from both directories; fixtures take precedence (inserted first).
    # A dict keyed by stem deduplicates: if the same filename exists in both dirs,
    # the fixture version wins.
    seen: dict[str, Path] = {}
    for source_dir in (FIXTURE_PDFS_DIR, PDFS_DIR):
        if source_dir.exists():
            for p in sorted(source_dir.glob("*.pdf")):
                seen.setdefault(p.stem, p)

    pdf_files = list(seen.values())
    if not pdf_files:
        print(f"No PDF files found in {FIXTURE_PDFS_DIR} or {PDFS_DIR}")
        return

    print(f"PDFs found    : {len(pdf_files)}")
    print(f"Fixture dir   : {FIXTURE_PDFS_DIR}")
    print(f"Real data dir : {PDFS_DIR}\n")

    for pdf_path in pdf_files:
        source_id = pdf_path.stem
        print(f"  {pdf_path.name}")
        result = asyncio.run(ingest_pdf(pdf_path, source_id))
        print(f"    source_id      : {result['source_id']}")
        print(f"    chunks_ingested: {result['chunks_ingested']}")
        print(f"    summary        : {result['summary'][:100]}{'...' if len(result['summary']) > 100 else ''}")
        print(f"    tags           : {result['tags']}")
        print()

    print(f"Done. {len(pdf_files)} PDF(s) ingested.")


if __name__ == "__main__":
    run_ingestion()
