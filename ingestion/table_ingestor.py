"""
ingestion/table_ingestor.py — Catalog and ingest a CSV or SQLite table.

Public API
----------
async ingest_table(file_path: Path, source_id: str, table_name: str | None = None) -> dict

Pipeline
--------
1. Detect file type from extension (.csv or .sqlite — ValueError otherwise).
2. Read schema and sample rows:
     CSV    — pandas DataFrame: column names + dtypes, up to 5 sample rows.
              table_name parameter is ignored.
     SQLite — PRAGMA table_info for columns, SELECT 5 rows for sample.
              Table selection rules:
                - Exactly one table in file → use it regardless of table_name.
                - Multiple tables, table_name is None → ValueError listing all tables.
                - Multiple tables, table_name provided → use it; ValueError if not found.
3. Send schema + sample to the LLM (using the 'planner' config slot) to
   generate a one-sentence summary, per-column descriptions, and relationship
   suggestions. Parse response with parse_llm_json.
4. Write index + detail entries to both manifest YAML files via
   write_source_to_manifest(), which also calls invalidate_manifest_cache().
   Detail entry schema matches manifest_detail.yaml (type, columns with
   name/type/description, relationships, row_count_approx; table_name for SQLite).
5. Return { source_id, row_count, columns, summary, table_name }.
   table_name is None for CSV sources.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from core.llm_config import get_llm
from core.parse import parse_llm_json
from ingestion.manifest_writer import write_source_to_manifest

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CATALOG_SYSTEM_PROMPT = (
    "You are cataloging a data table for a search system. "
    "Return JSON only: "
    '{ "summary": "one sentence describing the table", '
    '"column_descriptions": {"col_name": "description"}, '
    '"relationships": ["list of foreign key suggestions, e.g. col links to other_table.col"] }'
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dtype_to_str(dtype) -> str:
    """Convert a pandas dtype to a simple type string."""
    name = str(dtype)
    if "int" in name:
        return "integer"
    if "float" in name:
        return "float"
    if "bool" in name:
        return "boolean"
    if "datetime" in name:
        return "datetime"
    return "string"


def _read_csv_schema(file_path: Path) -> tuple[pd.DataFrame, list[dict], int]:
    """
    Read a CSV and return (df, columns_schema, row_count).
    columns_schema: list of { name, type } dicts.
    """
    df = pd.read_csv(file_path)
    columns = [
        {"name": col, "type": _dtype_to_str(df[col].dtype)}
        for col in df.columns
    ]
    return df, columns, len(df)


def _read_sqlite_schema(
    file_path: Path,
    requested_table: str | None,
) -> tuple[str, list[dict], list[dict], int]:
    """
    Read a SQLite file and return (table_name, columns_schema, sample_rows, row_count).
    columns_schema: list of { name, type } dicts (types from PRAGMA, lowercased).
    sample_rows: list of dicts (up to 5).

    Table selection:
      - 0 tables  → ValueError
      - 1 table   → use it unconditionally
      - N tables, requested_table is None  → ValueError listing all table names
      - N tables, requested_table provided → use it; ValueError if not in file
    """
    con = sqlite3.connect(file_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]

        if not tables:
            raise ValueError(f"No tables found in SQLite file: {file_path.name}")

        if len(tables) == 1:
            table_name = tables[0]
        elif requested_table is None:
            raise ValueError(
                f"'{file_path.name}' contains multiple tables: {tables}. "
                f"Specify table_name to select one."
            )
        elif requested_table not in tables:
            raise ValueError(
                f"Table '{requested_table}' not found in '{file_path.name}'. "
                f"Available tables: {tables}"
            )
        else:
            table_name = requested_table

        # Schema via PRAGMA
        cur.execute(f"PRAGMA table_info({table_name})")
        pragma_rows = cur.fetchall()
        columns = [
            {"name": row["name"], "type": (row["type"] or "TEXT").upper()}
            for row in pragma_rows
        ]

        # Normalise SQLite affinity types to simple strings
        type_map = {
            "INTEGER": "integer", "INT": "integer",
            "REAL": "float",    "NUMERIC": "float", "FLOAT": "float", "DOUBLE": "float",
            "TEXT": "string",   "VARCHAR": "string", "CHAR": "string",
            "BLOB": "string",   "BOOLEAN": "boolean",
            "DATE": "date",     "DATETIME": "datetime",
        }
        for col in columns:
            col["type"] = type_map.get(col["type"], col["type"].lower())

        # Row count
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cur.fetchone()[0]

        # Sample rows (up to 5)
        cur.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_rows = [dict(row) for row in cur.fetchall()]

    finally:
        con.close()

    return table_name, columns, sample_rows, row_count


def _build_schema_block(columns: list[dict], sample_rows: list[dict]) -> str:
    """Format columns + sample rows into a compact string for the LLM."""
    lines = ["Columns:"]
    for col in columns:
        lines.append(f"  {col['name']} ({col['type']})")
    lines.append("")
    lines.append("Sample rows (JSON):")
    for row in sample_rows:
        lines.append(f"  {row}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_table(
    file_path: Path,
    source_id: str,
    table_name: str | None = None,
) -> dict[str, Any]:
    """
    Catalog and ingest a CSV or SQLite table into both manifests.

    Parameters
    ----------
    file_path   : Absolute path to the .csv or .sqlite file.
    source_id   : Identifier used in the manifests. Must be unique across all sources.
    table_name  : SQLite only. If the file has exactly one table, ignored.
                  If the file has multiple tables and this is None, raises ValueError
                  listing all available table names. If provided, must exist in the file.
                  Ignored for CSV.

    Returns
    -------
    dict with keys: source_id, row_count, columns, summary, table_name
    table_name is None for CSV sources.
    """
    suffix = file_path.suffix.lower()
    if suffix not in (".csv", ".sqlite"):
        raise ValueError(
            f"Unsupported file type: '{suffix}'. Expected '.csv' or '.sqlite'."
        )

    # ── 1 & 2. Read schema and sample ────────────────────────────
    resolved_table: str | None = None
    if suffix == ".csv":
        df, columns, row_count = _read_csv_schema(file_path)
        sample_rows = df.head(min(5, len(df))).to_dict(orient="records")
        file_type = "csv"
    else:
        resolved_table, columns, sample_rows, row_count = _read_sqlite_schema(
            file_path, table_name
        )
        file_type = "sqlite"

    # ── 3. Catalog via LLM ────────────────────────────────────────
    schema_block = _build_schema_block(columns, sample_rows)
    llm = get_llm("planner")
    response = await llm.ainvoke([
        {"role": "system", "content": _CATALOG_SYSTEM_PROMPT},
        {"role": "user",   "content": schema_block},
    ])

    catalog = parse_llm_json(response.content)
    summary       = catalog["summary"]
    col_descs     = catalog.get("column_descriptions", {})
    relationships = catalog.get("relationships", [])

    # Merge LLM-generated descriptions into the columns list
    for col in columns:
        col["description"] = col_descs.get(col["name"], "")

    # ── 4. Write manifest entries ─────────────────────────────────
    name = file_path.stem.replace("_", " ").title()

    index_entry: dict = {
        "id":      source_id,
        "name":    name,
        "summary": summary,
    }
    detail_entry: dict = {
        "id":              source_id,
        "filename":        file_path.name,
        "type":            file_type,
        "base_path":       "data/tables/",
        "row_count_approx": row_count,
        "columns":         columns,
        "relationships":   relationships,
    }
    if resolved_table is not None:
        detail_entry["table_name"] = resolved_table

    write_source_to_manifest(source_id, index_entry, detail_entry)

    # ── 5. Return summary dict ────────────────────────────────────
    return {
        "source_id":  source_id,
        "row_count":  row_count,
        "columns":    [col["name"] for col in columns],
        "summary":    summary,
        "table_name": resolved_table,   # None for CSV
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m ingestion.table_ingestor
# ---------------------------------------------------------------------------

def _make_multi_table_sqlite(path: Path) -> None:
    """Create a SQLite file with two tables for multi-table tests."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("CREATE TABLE alpha (id INTEGER PRIMARY KEY, label TEXT)")
    cur.execute("INSERT INTO alpha VALUES (1, 'a'), (2, 'b')")
    cur.execute("CREATE TABLE beta (id INTEGER PRIMARY KEY, value REAL)")
    cur.execute("INSERT INTO beta VALUES (1, 9.9), (2, 8.8)")
    con.commit()
    con.close()


def test_table_ingestor():
    import tempfile

    from core.manifest import get_manifest_detail, get_manifest_index, invalidate_manifest_cache

    csv_path    = _PROJECT_ROOT / "data" / "tables" / "employees.csv"
    sqlite_path = _PROJECT_ROOT / "data" / "tables" / "salary_bands.sqlite"

    missing = [p for p in (csv_path, sqlite_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"SKIP: {p} not found — place the file in data/tables/ and retry")
        return

    # ── Test 1: ingest CSV — table_name is None in result ─────────
    print(f"Ingesting {csv_path.name} as source_id='employees' ...")
    csv_result = asyncio.run(ingest_table(csv_path, "employees"))
    print(f"Result: {csv_result}")

    assert csv_result["row_count"] > 0, f"Expected row_count > 0, got {csv_result['row_count']}"
    assert csv_result["summary"], "CSV summary must be non-empty"
    assert isinstance(csv_result["columns"], list) and len(csv_result["columns"]) > 0
    assert csv_result["table_name"] is None, "table_name must be None for CSV"
    print(f"PASS: CSV — row_count={csv_result['row_count']}, table_name=None")

    # ── Test 2: single-table SQLite — works without table_name ────
    print(f"\nIngesting {sqlite_path.name} as source_id='salary_bands' ...")
    sqlite_result = asyncio.run(ingest_table(sqlite_path, "salary_bands"))
    print(f"Result: {sqlite_result}")

    assert sqlite_result["row_count"] > 0
    assert sqlite_result["summary"], "SQLite summary must be non-empty"
    assert isinstance(sqlite_result["columns"], list) and len(sqlite_result["columns"]) > 0
    assert sqlite_result["table_name"] == "salary_bands", (
        f"Expected table_name='salary_bands', got '{sqlite_result['table_name']}'"
    )
    print(f"PASS: single-table SQLite — table_name='{sqlite_result['table_name']}'")

    # ── Test 3: manifest index updated for both sources ───────────
    invalidate_manifest_cache()
    index_str = get_manifest_index()
    assert "employees"    in index_str
    assert "salary_bands" in index_str
    assert csv_result["summary"]    in index_str
    assert sqlite_result["summary"] in index_str
    print("PASS: manifest index contains both source_ids and LLM-generated summaries")

    # ── Test 4: manifest detail correct schema ────────────────────
    csv_detail = get_manifest_detail("employees")
    assert "CSV"      in csv_detail and "Columns:" in csv_detail
    print("PASS: manifest detail for CSV has correct schema fields")

    sqlite_detail = get_manifest_detail("salary_bands")
    assert "SQLITE"   in sqlite_detail and "Columns:" in sqlite_detail
    assert "salary_bands" in sqlite_detail
    print("PASS: manifest detail for SQLite has correct schema fields")

    # ── Tests 5-8: multi-table SQLite scenarios ───────────────────
    with tempfile.TemporaryDirectory() as tmp:
        multi_db = Path(tmp) / "multi.sqlite"
        _make_multi_table_sqlite(multi_db)

        # Test 5: no table_name → ValueError listing both tables
        try:
            asyncio.run(ingest_table(multi_db, "multi_no_name"))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            msg = str(exc)
            assert "alpha" in msg and "beta" in msg, (
                f"Error must list available tables, got: {msg}"
            )
            assert "table_name" in msg
            print(f"PASS: multi-table + no table_name raises ValueError: {msg}")

        # Test 6: valid table_name → ingests correct table
        result_alpha = asyncio.run(ingest_table(multi_db, "multi_alpha", table_name="alpha"))
        assert result_alpha["table_name"] == "alpha"
        assert result_alpha["row_count"]  == 2
        assert "id" in result_alpha["columns"] and "label" in result_alpha["columns"]
        print(f"PASS: multi-table + table_name='alpha' ingests correct table")

        result_beta = asyncio.run(ingest_table(multi_db, "multi_beta", table_name="beta"))
        assert result_beta["table_name"] == "beta"
        assert result_beta["row_count"]  == 2
        assert "value" in result_beta["columns"]
        print(f"PASS: multi-table + table_name='beta' ingests correct table")

        # Test 7: invalid table_name → ValueError naming the bad table
        try:
            asyncio.run(ingest_table(multi_db, "multi_bad", table_name="gamma"))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            msg = str(exc)
            assert "gamma" in msg, f"Error must mention the bad table name, got: {msg}"
            assert "alpha" in msg and "beta" in msg, (
                f"Error must list available tables, got: {msg}"
            )
            print(f"PASS: invalid table_name raises ValueError: {msg}")

    # ── Test 8: unsupported extension raises ValueError ───────────
    try:
        asyncio.run(ingest_table(Path("report.xlsx"), "bad_source"))
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert ".xlsx" in str(exc)
        print(f"PASS: ValueError raised for unsupported extension: {exc}")

    print("\nPASS: all table_ingestor tests passed")


if __name__ == "__main__":
    test_table_ingestor()
