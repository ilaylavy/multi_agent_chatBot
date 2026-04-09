"""
core/manifest.py — Manifest loader and formatter.

Two public functions:
  get_manifest_index()            → formatted string for the Planner prompt
  get_manifest_detail(source_id)  → formatted string for a Worker prompt

Both cache their YAML reads in a module-level dict. Call
invalidate_manifest_cache() to clear the cache and force the next read to
reload from disk — useful after running an ingestion script that adds new
sources without restarting the server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dict-based cache — keys: "index", "detail"
# Replaced lru_cache so the cache can be explicitly cleared at runtime.
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def invalidate_manifest_cache() -> None:
    """
    Clear the manifest cache.

    The next call to get_manifest_index() or get_manifest_detail() will
    re-read the YAML files from disk. No-op if the cache is already empty.
    """
    _cache.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_paths() -> dict:
    """Return the paths block from config.yaml."""
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["paths"]


def _load_index_raw() -> dict:
    """Return the parsed manifest_index.yaml, reading from disk on cache miss."""
    if "index" not in _cache:
        path = _PROJECT_ROOT / _config_paths()["manifest_index"]
        with open(path, "r") as f:
            _cache["index"] = yaml.safe_load(f)
    return _cache["index"]


def _load_detail_raw() -> dict:
    """Return the parsed manifest_detail.yaml, reading from disk on cache miss."""
    if "detail" not in _cache:
        path = _PROJECT_ROOT / _config_paths()["manifest_detail"]
        with open(path, "r") as f:
            _cache["detail"] = yaml.safe_load(f)
    return _cache["detail"]


def _format_index(raw: dict) -> str:
    """Convert the raw index dict into a Planner-ready string."""
    lines: list[str] = ["AVAILABLE DATA SOURCES", "=" * 22, ""]

    if raw.get("pdfs"):
        lines.append("PDFs:")
        for entry in raw["pdfs"]:
            lines.append(f"  - id: {entry['id']}")
            lines.append(f"    name: {entry['name']}")
            lines.append(f"    summary: {entry['summary']}")
            contains = entry.get("contains", [])
            if contains:
                lines.append(f"    contains: {', '.join(contains)}")
            lines.append("")

    if raw.get("tables"):
        lines.append("Tables:")
        for entry in raw["tables"]:
            lines.append(f"  - id: {entry['id']}")
            lines.append(f"    name: {entry['name']}")
            lines.append(f"    summary: {entry['summary']}")
            contains = entry.get("contains", [])
            if contains:
                lines.append(f"    contains: {', '.join(contains)}")
            lines.append("")

    return "\n".join(lines).rstrip()


def _format_pdf_detail(entry: dict[str, Any]) -> str:
    """Format a PDF detail entry for a Librarian prompt."""
    lines: list[str] = [
        f"SOURCE DETAIL: {entry['id']}",
        "=" * 40,
        f"Type:     PDF",
        f"Filename: {entry['filename']}",
        f"Pages:    {entry['pages']}",
        "",
        "Sections:",
    ]
    for section in entry.get("sections", []):
        lines.append(f"  - {section}")

    tags = entry.get("tags", [])
    if tags:
        lines.append("")
        lines.append(f"Tags: {', '.join(tags)}")

    notes = entry.get("notes")
    if notes:
        lines.append("")
        lines.append(f"Notes: {notes}")

    return "\n".join(lines)


def _format_table_detail(entry: dict[str, Any]) -> str:
    """Format a table detail entry for a Data Scientist prompt."""
    lines: list[str] = [
        f"SOURCE DETAIL: {entry['id']}",
        "=" * 40,
        f"Type:     {entry['type'].upper()}",
        f"Filename: {entry['filename']}",
    ]

    if "table_name" in entry:
        lines.append(f"Table name: {entry['table_name']}")

    lines.append(f"Approx rows: {entry.get('row_count_approx', 'unknown')}")
    lines.append("")
    lines.append("Columns:")

    for col in entry.get("columns", []):
        col_line = f"  - {col['name']} ({col['type']}): {col['description']}"
        samples = col.get("sample_values")
        if samples:
            col_line += f"  [e.g. {', '.join(str(s) for s in samples)}]"
        lines.append(col_line)

    relationships = entry.get("relationships", [])
    if relationships:
        lines.append("")
        lines.append("Relationships:")
        for rel in relationships:
            lines.append(f"  - {rel}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_manifest_index_raw() -> dict:
    """Return the parsed manifest_index.yaml as a plain dict."""
    return _load_index_raw()


def get_manifest_detail_raw() -> dict:
    """Return the parsed manifest_detail.yaml as a plain dict."""
    return _load_detail_raw()


def get_manifest_index() -> str:
    """
    Return the full manifest index as a formatted string.

    Ready to paste directly into a Planner prompt.
    Reads manifest_index.yaml (path from config.yaml) and caches the parse.
    Call invalidate_manifest_cache() to force a re-read.
    """
    return _format_index(_load_index_raw())


def get_manifest_detail(source_id: str) -> str:
    """
    Return the detail entry for source_id as a formatted string.

    Ready to paste directly into a Worker (Librarian / Data Scientist) prompt.
    Searches both the 'pdfs' and 'tables' sections of manifest_detail.yaml.

    Parameters
    ----------
    source_id : str
        The id value as declared in manifest_detail.yaml.

    Returns
    -------
    str
        Formatted detail block for that source.

    Raises
    ------
    ValueError
        If source_id is not found in either pdfs or tables sections.
    """
    raw = _load_detail_raw()

    for entry in raw.get("pdfs", []):
        if entry["id"] == source_id:
            return _format_pdf_detail(entry)

    for entry in raw.get("tables", []):
        if entry["id"] == source_id:
            return _format_table_detail(entry)

    all_ids = (
        [e["id"] for e in raw.get("pdfs", [])]
        + [e["id"] for e in raw.get("tables", [])]
    )
    raise ValueError(
        f"source_id '{source_id}' not found in manifest_detail.yaml. "
        f"Known IDs: {all_ids}"
    )


# ---------------------------------------------------------------------------
# Isolated test — run with: python core/manifest.py
# ---------------------------------------------------------------------------

def test_manifest():
    # 1. Index — print full output for visual inspection
    print("=== get_manifest_index() ===\n")
    index_str = get_manifest_index()
    print(index_str)
    assert "travel_policy_2024" in index_str
    assert "employees" in index_str
    assert "PDFs:" in index_str
    assert "Tables:" in index_str

    # 2. PDF detail
    print("\n\n=== get_manifest_detail('travel_policy_2024') ===\n")
    pdf_detail = get_manifest_detail("travel_policy_2024")
    print(pdf_detail)
    assert "travel_policy_2024" in pdf_detail
    assert "PDF" in pdf_detail
    assert "Sections:" in pdf_detail

    # 3. Table detail
    print("\n\n=== get_manifest_detail('employees') ===\n")
    table_detail = get_manifest_detail("employees")
    print(table_detail)
    assert "employees" in table_detail
    assert "CSV" in table_detail
    assert "Columns:" in table_detail

    # 4. Unknown source_id must raise ValueError with a helpful message
    print("\n\n=== get_manifest_detail('fake_source_xyz') — expect ValueError ===\n")
    try:
        get_manifest_detail("fake_source_xyz")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        print(f"Correctly raised ValueError: {exc}")
        assert "fake_source_xyz" in str(exc)
        assert "Known IDs:" in str(exc)

    # 5. Cache invalidation — confirm invalidate_manifest_cache() forces a re-read
    print("\n\n=== invalidate_manifest_cache() — confirm re-read from disk ===\n")

    # Start clean so the assertions below are deterministic
    invalidate_manifest_cache()
    assert _cache == {}, "Cache should be empty after invalidate_manifest_cache()"

    # Populate the cache
    get_manifest_index()
    get_manifest_detail("employees")
    assert "index"  in _cache, "Cache should contain 'index' after get_manifest_index()"
    assert "detail" in _cache, "Cache should contain 'detail' after get_manifest_detail()"
    print("PASS: cache populated after reads")

    # Invalidate — cache must be empty immediately
    invalidate_manifest_cache()
    assert _cache == {}, "Cache should be empty immediately after invalidate_manifest_cache()"
    print("PASS: cache empty after invalidate_manifest_cache()")

    # Next read must re-populate from disk and return correct data
    re_index = get_manifest_index()
    assert "index" in _cache,               "Cache should be re-populated after next get_manifest_index()"
    assert "travel_policy_2024" in re_index, "Re-read index must contain expected source"
    print("PASS: next get_manifest_index() re-reads from disk and returns correct data")

    print("\nPASS: all manifest tests passed")


if __name__ == "__main__":
    test_manifest()
