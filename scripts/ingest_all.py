"""
scripts/ingest_all.py — Ingest all PDFs and tables into the system.

Scans two directory pairs (fixtures first, then real data):
  PDFs:   tests/fixtures/pdfs/  →  data/pdfs/
  Tables: tests/fixtures/tables/ →  data/tables/

Deduplicates by filename stem (fixtures take precedence).

For each PDF:  calls ingest_pdf()  → parses, catalogs via LLM, chunks into ChromaDB, writes manifests
For each table: calls ingest_table() → reads schema, catalogs via LLM, writes manifests

Run with: python -m scripts.ingest_all
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from ingestion.manifest_writer import regenerate_data_context
from ingestion.pdf_ingestor import ingest_pdf
from ingestion.relationship_detector import detect_relationships
from ingestion.table_ingestor import ingest_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIXTURE_PDFS_DIR   = PROJECT_ROOT / "tests" / "fixtures" / "pdfs"
FIXTURE_TABLES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "tables"
PDFS_DIR           = PROJECT_ROOT / "data" / "pdfs"
TABLES_DIR         = PROJECT_ROOT / "data" / "tables"


def _collect_files(fixture_dir: Path, data_dir: Path, extensions: list[str]) -> list[Path]:
    """Collect files from fixture and data dirs, deduped by stem (fixtures win)."""
    seen: dict[str, Path] = {}
    for source_dir in (fixture_dir, data_dir):
        if source_dir.exists():
            for ext in extensions:
                for p in sorted(source_dir.glob(f"*.{ext}")):
                    seen.setdefault(p.stem, p)
    return list(seen.values())


def _list_sqlite_tables(file_path: Path) -> list[str]:
    """Return user table names from a SQLite file."""
    conn = sqlite3.connect(str(file_path))
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
    finally:
        conn.close()
    return tables


def _ingest_pdfs(pdf_files: list[Path]) -> int:
    """Ingest all PDFs. Returns count ingested."""
    if not pdf_files:
        print("  No PDF files found.\n")
        return 0

    print(f"  Found {len(pdf_files)} PDF(s)\n")
    for pdf_path in pdf_files:
        source_id = pdf_path.stem
        print(f"  {pdf_path.name}")
        result = asyncio.run(ingest_pdf(pdf_path, source_id, regenerate_context=False))
        print(f"    source_id      : {result['source_id']}")
        print(f"    chunks_ingested: {result['chunks_ingested']}")
        summary = result["summary"]
        print(f"    summary        : {summary[:100]}{'...' if len(summary) > 100 else ''}")
        print(f"    tags           : {result['tags']}")
        print()

    return len(pdf_files)


def _ingest_tables(table_files: list[Path]) -> int:
    """Ingest all tables (CSV and SQLite). Returns count ingested."""
    if not table_files:
        print("  No table files found.\n")
        return 0

    print(f"  Found {len(table_files)} table file(s)\n")
    count = 0

    for table_path in table_files:
        if table_path.suffix == ".sqlite":
            tables = _list_sqlite_tables(table_path)
            for table_name in tables:
                source_id = table_name if len(tables) == 1 else f"{table_path.stem}_{table_name}"
                print(f"  {table_path.name} -> {table_name}")
                result = asyncio.run(ingest_table(
                    table_path, source_id, table_name=table_name,
                    regenerate_context=False,
                ))
                _print_table_result(result)
                count += 1
        else:
            source_id = table_path.stem
            print(f"  {table_path.name}")
            result = asyncio.run(ingest_table(table_path, source_id, regenerate_context=False))
            _print_table_result(result)
            count += 1

    return count


def _print_table_result(result: dict) -> None:
    """Print ingestion result for a single table."""
    print(f"    source_id : {result['source_id']}")
    print(f"    row_count : {result['row_count']}")
    print(f"    columns   : {result['columns']}")
    summary = result["summary"]
    print(f"    summary   : {summary[:100]}{'...' if len(summary) > 100 else ''}")
    print()


def run_ingestion() -> None:
    pdf_files = _collect_files(FIXTURE_PDFS_DIR, PDFS_DIR, ["pdf"])
    table_files = _collect_files(FIXTURE_TABLES_DIR, TABLES_DIR, ["csv", "sqlite"])

    print(f"Fixture dirs : {FIXTURE_PDFS_DIR}")
    print(f"               {FIXTURE_TABLES_DIR}")
    print(f"Data dirs    : {PDFS_DIR}")
    print(f"               {TABLES_DIR}")
    print()

    print("=" * 60)
    print("PDFS")
    print("=" * 60)
    n_pdfs = _ingest_pdfs(pdf_files)

    print("=" * 60)
    print("TABLES")
    print("=" * 60)
    n_tables = _ingest_tables(table_files)

    print("=" * 60)
    print("RELATIONSHIP DETECTION")
    print("=" * 60)
    rel_result = detect_relationships()
    n_cross = len(rel_result["cross_source"])
    n_per_table = sum(1 for r in rel_result["per_table"].values() if r)
    print(f"  Cross-source relationships: {n_cross}")
    print(f"  Tables with relationships:  {n_per_table}")
    for rel in rel_result["cross_source"]:
        sources = rel["sources"]
        print(f"    {sources[0]} <-> {sources[1]} via {rel['shared_key']} (verified: {rel['verified']})")
    print()

    print("=" * 60)
    print("DATA CONTEXT")
    print("=" * 60)
    paragraph = asyncio.run(regenerate_data_context())
    print(f"  {paragraph}")
    print()

    print(f"Done. {n_pdfs} PDF(s) and {n_tables} table(s) ingested, {n_cross} relationships detected.")


if __name__ == "__main__":
    run_ingestion()
