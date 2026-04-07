"""
ingestion/pdf_ingestor.py — Parse, catalog, chunk, and ingest a PDF.

Public API
----------
async ingest_pdf(file_path: Path, source_id: str) -> dict

Pipeline
--------
1. Parse PDF pages with pymupdf4llm.
2. Send the first 3000 characters to the LLM (using the 'planner' config slot)
   to generate a one-sentence summary, section headings, and tags.
3. Write index + detail entries to both manifest YAML files via
   write_source_to_manifest(), which also calls invalidate_manifest_cache().
4. Chunk the full document text using chunk_size and chunk_overlap from
   config.yaml → retrieval section.
5. Delete any existing ChromaDB collection for this source_id, then ingest all
   chunks as a fresh collection. Collection name == source_id.
6. Return { source_id, chunks_ingested, summary, tags }.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import chromadb
import pymupdf4llm

from core.llm_config import _load_config, get_llm
from core.parse import parse_llm_json
from core.retriever import ChromaRetriever
from ingestion.manifest_writer import write_source_to_manifest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CATALOG_SYSTEM_PROMPT = (
    "You are reading a document to catalog it for a search system. "
    "Return JSON only: "
    '{ "summary": "one sentence", '
    '"sections": ["list of main headings"], '
    '"tags": ["5-10 keywords"] }'
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
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
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_pdf(file_path: Path, source_id: str) -> dict[str, Any]:
    """
    Parse, catalog, chunk, and ingest a PDF into ChromaDB and both manifests.

    Parameters
    ----------
    file_path : Absolute path to the PDF file.
    source_id : Identifier used in the manifests and as the ChromaDB collection
                name. Must be unique across all sources.

    Returns
    -------
    dict with keys: source_id, chunks_ingested, summary, tags
    """
    # ── 1. Parse PDF ─────────────────────────────────────────────
    pages: list[dict] = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    page_count = len(pages)
    full_text = "\n\n".join(page.get("text", "") for page in pages)

    # ── 2. Catalog via LLM ────────────────────────────────────────
    preview = full_text[:3000]
    llm = get_llm("planner")
    response = await llm.ainvoke([
        {"role": "system", "content": _CATALOG_SYSTEM_PROMPT},
        {"role": "user",   "content": preview},
    ])

    catalog   = parse_llm_json(response.content)
    summary   = catalog["summary"]
    sections  = catalog.get("sections", [])
    tags      = catalog.get("tags", [])

    # ── 3. Write manifest entries ──────────────────────────────────
    # name: derive from the filename stem, title-cased
    name = file_path.stem.replace("_", " ").title()

    index_entry: dict = {
        "id":      source_id,
        "name":    name,
        "summary": summary,
    }
    detail_entry: dict = {
        "id":       source_id,
        "filename": file_path.name,
        "type":     "pdf",
        "pages":    page_count,
        "sections": sections,
        "tags":     tags,
    }
    write_source_to_manifest(source_id, index_entry, detail_entry)

    # ── 4. Chunk full text ────────────────────────────────────────
    retrieval_cfg = _load_config()["retrieval"]
    chunk_size    = retrieval_cfg["chunk_size"]
    chunk_overlap = retrieval_cfg["chunk_overlap"]

    ids:       list[str]  = []
    texts:     list[str]  = []
    metadatas: list[dict] = []
    global_chunk_index = 0

    for page in pages:
        page_number = page.get("metadata", {}).get("page", 0) + 1  # 1-indexed
        page_text   = page.get("text", "")
        page_chunks = _chunk_text(page_text, chunk_size, chunk_overlap)

        for chunk in page_chunks:
            ids.append(f"{source_id}_{global_chunk_index}")
            texts.append(chunk)
            metadatas.append({
                "source_pdf":  file_path.name,
                "page_number": page_number,
                "chunk_index": global_chunk_index,
            })
            global_chunk_index += 1

    # ── 5. Ingest into ChromaDB ───────────────────────────────────
    retriever = ChromaRetriever()
    client: chromadb.ClientAPI = retriever._client

    # get_or_create keeps the collection alive during reingest — no deletion gap.
    # Then clear any previously stored documents by their IDs so a reingest
    # with fewer chunks doesn't leave stale entries behind.
    collection = client.get_or_create_collection(source_id)
    existing = collection.get(include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    # ── 6. Return summary dict ────────────────────────────────────
    return {
        "source_id":       source_id,
        "chunks_ingested": len(ids),
        "summary":         summary,
        "tags":            tags,
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m ingestion.pdf_ingestor
# ---------------------------------------------------------------------------

def test_pdf_ingestor():
    from core.manifest import get_manifest_detail, get_manifest_index, invalidate_manifest_cache

    pdf_path = _PROJECT_ROOT / "data" / "pdfs" / "travel_policy_2024.pdf"
    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found — place the PDF in data/pdfs/ and retry")
        return

    print(f"Ingesting {pdf_path.name} as source_id='travel_policy_2024' ...")
    result = asyncio.run(ingest_pdf(pdf_path, "travel_policy_2024"))

    print(f"Result: {result}")

    # ── chunks_ingested > 0 ───────────────────────────────────────
    assert result["chunks_ingested"] > 0, (
        f"Expected at least one chunk, got {result['chunks_ingested']}"
    )
    print(f"PASS: chunks_ingested = {result['chunks_ingested']}")

    # ── summary is non-empty ──────────────────────────────────────
    assert result["summary"], "summary must be a non-empty string"
    print(f"PASS: summary = '{result['summary']}'")

    # ── tags is a non-empty list ──────────────────────────────────
    assert isinstance(result["tags"], list) and len(result["tags"]) > 0, (
        "tags must be a non-empty list"
    )
    print(f"PASS: tags = {result['tags']}")

    # ── manifest index was written ────────────────────────────────
    invalidate_manifest_cache()
    index_str = get_manifest_index()
    assert "travel_policy_2024" in index_str, (
        "travel_policy_2024 must appear in manifest index after ingest"
    )
    assert result["summary"] in index_str, (
        "LLM-generated summary must appear in manifest index"
    )
    print("PASS: manifest index contains source_id and LLM-generated summary")

    # ── manifest detail was written ───────────────────────────────
    detail_str = get_manifest_detail("travel_policy_2024")
    assert "PDF" in detail_str,          "detail must identify source type as PDF"
    assert "travel_policy_2024" in detail_str
    assert "Sections:" in detail_str,    "detail must include sections from LLM catalog"
    print("PASS: manifest detail contains correct schema fields")

    # ── ChromaDB collection is queryable ──────────────────────────
    retriever = ChromaRetriever()
    chunks = asyncio.run(
        retriever.search(
            query="Business Class flight entitlement",
            source_id="travel_policy_2024",
            top_k=3,
        )
    )
    assert len(chunks) > 0, "ChromaDB search must return at least one chunk after ingest"
    assert all(c["chunk_text"] for c in chunks), "All returned chunks must have text"
    print(f"PASS: ChromaDB search returned {len(chunks)} chunk(s) after ingest")

    # ── Reingest: no exception, same chunk count, no duplication ─
    print("\nReingesting same source_id to verify idempotency ...")
    result2 = asyncio.run(ingest_pdf(pdf_path, "travel_policy_2024"))

    assert result2["chunks_ingested"] == result["chunks_ingested"], (
        f"Second ingest produced {result2['chunks_ingested']} chunks, "
        f"expected {result['chunks_ingested']}"
    )
    print(f"PASS: chunks_ingested matches on reingest ({result2['chunks_ingested']})")

    # Confirm ChromaDB holds exactly N chunks — not 2N
    client = retriever._client
    collection = client.get_collection("travel_policy_2024")
    stored_count = collection.count()
    assert stored_count == result["chunks_ingested"], (
        f"ChromaDB contains {stored_count} chunks after reingest, "
        f"expected {result['chunks_ingested']} (not doubled)"
    )
    print(f"PASS: ChromaDB contains exactly {stored_count} chunk(s) after reingest — not doubled")

    print("\nPASS: all pdf_ingestor tests passed")


if __name__ == "__main__":
    test_pdf_ingestor()
