"""
ingestion/relationship_detector.py — Detect cross-source relationships.

Scans all table entries in manifest_detail.yaml, finds columns with
matching names or FK patterns across tables, verifies them against
actual data, and writes:
  - Per-table structured relationships to manifest_detail.yaml
  - Cross-source relationships to manifest_index.yaml

Public API
----------
detect_relationships(*, detail_path=None, index_path=None) -> dict
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.manifest_writer import (
    _read_yaml,
    update_table_relationships,
    write_cross_source_relationships,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_column_values(
    base_path: str,
    filename: str,
    file_type: str,
    col_name: str,
    table_name: str | None = None,
) -> set:
    """Load distinct non-null values for a column from a CSV or SQLite file."""
    file_path = _PROJECT_ROOT / base_path / filename
    if not file_path.exists():
        return set()

    if file_type == "csv":
        try:
            df = pd.read_csv(file_path)
            if col_name in df.columns:
                return set(df[col_name].dropna().unique().tolist())
        except Exception:
            logger.warning("Failed to read %s for relationship detection", file_path)
        return set()

    if file_type == "sqlite":
        try:
            conn = sqlite3.connect(file_path)
            tbl = table_name or filename.replace(".sqlite", "")
            cur = conn.cursor()
            cur.execute(f"SELECT DISTINCT [{col_name}] FROM [{tbl}] WHERE [{col_name}] IS NOT NULL")
            values = {row[0] for row in cur.fetchall()}
            conn.close()
            return values
        except Exception:
            logger.warning("Failed to read %s for relationship detection", file_path)
        return set()

    return set()


def _strip_prefix(col_name: str) -> str:
    """
    Strip common FK prefixes to find a base column name.

    e.g. 'lead_employee_id' -> 'employee_id', 'manager_id' -> 'id'
    """
    # Common patterns: prefix_<base_col>
    # We try to match against other tables' columns both with and without prefix
    return col_name


def _find_column_matches(
    tables: list[dict],
) -> list[tuple[dict, str, dict, str]]:
    """
    Find pairs of (table_a, col_a, table_b, col_b) that might be related.

    Matches on:
    1. Exact column name match (e.g. employee_id in both tables)
    2. FK pattern: col_a ends with _<col_b_name> (e.g. lead_employee_id -> employee_id)
    """
    matches = []
    for i, table_a in enumerate(tables):
        cols_a = {c["name"]: c for c in table_a.get("columns", [])}
        for j, table_b in enumerate(tables):
            if j <= i:
                continue
            cols_b = {c["name"]: c for c in table_b.get("columns", [])}

            # Exact name matches (skip generic 'id' — too many false positives)
            shared = set(cols_a.keys()) & set(cols_b.keys())
            for col_name in shared:
                if col_name == "id":
                    continue
                matches.append((table_a, col_name, table_b, col_name))

            # FK pattern: col in A ends with _<col_name in B>
            # e.g. lead_employee_id in projects -> employee_id in employees
            # Skip matches against bare 'id' — too generic
            for ca in cols_a:
                for cb in cols_b:
                    if ca == cb:
                        continue
                    if cb != "id" and ca.endswith(f"_{cb}"):
                        matches.append((table_a, ca, table_b, cb))
                    if ca != "id" and cb.endswith(f"_{ca}"):
                        matches.append((table_a, ca, table_b, cb))

    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_relationships(
    *,
    detail_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """
    Detect and write cross-source relationships between tables.

    Scans manifest_detail.yaml for column matches, verifies against actual
    data, and writes structured relationships to both manifest files.

    Returns
    -------
    dict with keys: per_table (dict of source_id -> relationships),
                    cross_source (list of cross-source relationship dicts)
    """
    from core.llm_config import _load_config
    paths = _load_config()["paths"]
    det_path = detail_path or (_PROJECT_ROOT / paths["manifest_detail"])
    idx_path = index_path or (_PROJECT_ROOT / paths["manifest_index"])

    det_data = _read_yaml(det_path)
    tables = det_data.get("tables", [])

    if len(tables) < 2:
        logger.info("Fewer than 2 tables — skipping relationship detection")
        return {"per_table": {}, "cross_source": []}

    # Find candidate matches
    candidates = _find_column_matches(tables)

    # Per-table relationships: source_id -> list of structured rel dicts
    per_table: dict[str, list[dict]] = {t["id"]: [] for t in tables}
    # Cross-source relationships for the index
    cross_source: list[dict] = []
    # Track already-added pairs to avoid duplicates
    seen_pairs: set[tuple] = set()

    for table_a, col_a, table_b, col_b in candidates:
        pair_key = tuple(sorted([(table_a["id"], col_a), (table_b["id"], col_b)]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Load actual values to verify
        vals_a = _load_column_values(
            table_a.get("base_path", "data/tables/"),
            table_a["filename"],
            table_a["type"],
            col_a,
            table_a.get("table_name"),
        )
        vals_b = _load_column_values(
            table_b.get("base_path", "data/tables/"),
            table_b["filename"],
            table_b["type"],
            col_b,
            table_b.get("table_name"),
        )

        if not vals_a or not vals_b:
            continue

        # Check subset relationship in either direction
        a_subset_b = vals_a.issubset(vals_b)
        b_subset_a = vals_b.issubset(vals_a)
        overlap = bool(vals_a & vals_b)
        verified = a_subset_b or b_subset_a

        if not overlap:
            continue

        # Determine direction: the "referring" side is the subset
        if a_subset_b:
            from_table, from_col = table_a, col_a
            to_table, to_col = table_b, col_b
        else:
            from_table, from_col = table_b, col_b
            to_table, to_col = table_a, col_a

        # Add per-table relationship
        per_table[from_table["id"]].append({
            "from_column": from_col,
            "to_table": to_table["id"],
            "to_column": to_col,
            "verified": verified,
        })

        # If bidirectional subset (same values), add reverse too
        if a_subset_b and b_subset_a and from_table["id"] != to_table["id"]:
            per_table[to_table["id"]].append({
                "from_column": to_col,
                "to_table": from_table["id"],
                "to_column": from_col,
                "verified": verified,
            })

        # Cross-source relationship
        shared_key = col_a if col_a == col_b else f"{col_a}/{col_b}"
        cross_source.append({
            "sources": sorted([table_a["id"], table_b["id"]]),
            "shared_key": shared_key,
            "description": f"tables share joinable key ({shared_key})",
            "verified": verified,
        })

    # Write per-table relationships
    for source_id, rels in per_table.items():
        if rels:
            update_table_relationships(source_id, rels, detail_path=det_path)

    # Write cross-source relationships
    if cross_source:
        write_cross_source_relationships(cross_source, index_path=idx_path)

    logger.info(
        "Relationship detection complete: %d cross-source, %d tables with relationships",
        len(cross_source),
        sum(1 for r in per_table.values() if r),
    )

    return {"per_table": per_table, "cross_source": cross_source}
