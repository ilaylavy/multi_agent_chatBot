"""
core/manifest_prefilter.py — RAG-based pre-filter over the manifest index.

Embeds each source's metadata (summary + contains) into a dedicated ChromaDB
collection (source_index).  At query time, retrieves the top-K most relevant
sources and passes only those to the Planner.

Public functions:
  build_source_index()                        → populate/rebuild the source_index collection
  prefilter_manifest(query)                   → (filtered_manifest_str, trace_list)
  get_last_prefilter_trace(session_id)        → trace list stored by the most recent call
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import chromadb
import yaml

from core.manifest import (
    format_manifest_index,
    get_manifest_index,
    get_manifest_index_raw,
    register_invalidation_callback,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COLLECTION_NAME = "source_index"

# ---------------------------------------------------------------------------
# Stale flag — set True whenever the manifest cache is invalidated.
# The next prefilter_manifest() call will rebuild before querying.
# ---------------------------------------------------------------------------

_source_index_stale: bool = True


def _mark_stale() -> None:
    global _source_index_stale
    _source_index_stale = True


register_invalidation_callback(_mark_stale)

# ---------------------------------------------------------------------------
# Per-session trace store — planner_node writes, api.py reads
# ---------------------------------------------------------------------------

_prefilter_traces: dict[str, list[dict]] = {}


def get_last_prefilter_trace(session_id: str) -> list[dict] | None:
    """Return the prefilter trace for *session_id*, or None if not set."""
    return _prefilter_traces.get(session_id)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _config() -> dict:
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _chroma_path() -> str:
    return str(_PROJECT_ROOT / _config()["paths"]["chroma_db"])


def _prefilter_top_k() -> int:
    return _config().get("retrieval", {}).get("prefilter_top_k", 5)


# ---------------------------------------------------------------------------
# Build / rebuild the source_index collection
# ---------------------------------------------------------------------------

def build_source_index(*, chroma_path: str | None = None) -> None:
    """
    (Re)build the source_index ChromaDB collection from the current manifest.

    Deletes and recreates the collection so it is always in sync with
    manifest_index.yaml.  With 8–50 sources this takes milliseconds.
    """
    global _source_index_stale

    path = chroma_path or _chroma_path()
    client = chromadb.PersistentClient(path=path)

    # Delete existing collection if present
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    collection = client.create_collection(name=_COLLECTION_NAME)

    raw = get_manifest_index_raw()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for section in ("pdfs", "tables"):
        for entry in raw.get(section, []):
            source_id = entry["id"]
            name = entry.get("name", source_id)
            summary = entry.get("summary", "")
            contains = entry.get("contains", [])
            kind = entry.get("kind", "record" if section == "tables" else "policy")

            text = f"{name}. {summary}. Contains: {', '.join(contains)}"
            ids.append(source_id)
            documents.append(text)
            metadatas.append({"kind": kind})

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("source_index rebuilt with %d sources", len(ids))
    else:
        logger.warning("source_index rebuilt but manifest had no sources")

    _source_index_stale = False


# ---------------------------------------------------------------------------
# Query-time pre-filter
# ---------------------------------------------------------------------------

def prefilter_manifest(
    query: str,
    *,
    chroma_path: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Return a filtered manifest and a trace list for the given *query*.

    Uses vector similarity over source metadata to select the most relevant
    sources, then expands via relationships and enforces kind diversity.

    Falls back to the full manifest when fewer than 3 sources are selected
    or the source_index collection does not exist.
    """
    global _source_index_stale

    # Lazy rebuild if stale
    if _source_index_stale:
        build_source_index(chroma_path=chroma_path)

    path = chroma_path or _chroma_path()
    client = chromadb.PersistentClient(path=path)

    try:
        collection = client.get_collection(name=_COLLECTION_NAME)
    except Exception:
        logger.warning("source_index collection not found — using full manifest")
        return get_manifest_index(), []

    count = collection.count()
    if count == 0:
        logger.warning("source_index collection is empty — using full manifest")
        return get_manifest_index(), []

    top_k = _prefilter_top_k()
    n = min(top_k, count)
    results = collection.query(query_texts=[query], n_results=n)

    # Also fetch ALL results for diversity fallback (cheap — small collection)
    all_results = collection.query(query_texts=[query], n_results=count)

    # Build selected set and trace from top-K results
    selected_ids: set[str] = set()
    trace: list[dict] = []
    # Map source_id -> score for diversity fallback
    all_scores: dict[str, float] = {}
    all_kinds: dict[str, str] = {}

    for i, source_id in enumerate(all_results["ids"][0]):
        distance = all_results["distances"][0][i]
        score = round(1 / (1 + distance), 4)
        all_scores[source_id] = score
        meta = all_results["metadatas"][0][i] if all_results["metadatas"] else {}
        all_kinds[source_id] = meta.get("kind", "")

    for i, source_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = round(1 / (1 + distance), 4)
        selected_ids.add(source_id)
        trace.append({
            "source_id": source_id,
            "score": score,
            "expanded_via_relationship": False,
        })

    # --- Relationship expansion (one-hop) ---
    raw = get_manifest_index_raw()
    relationships = raw.get("relationships", [])
    for rel in relationships:
        sources = rel.get("sources", [])
        if len(sources) != 2:
            continue
        a, b = sources
        if a in selected_ids and b not in selected_ids:
            selected_ids.add(b)
            trace.append({
                "source_id": b,
                "score": all_scores.get(b, 0.0),
                "expanded_via_relationship": True,
            })
        elif b in selected_ids and a not in selected_ids:
            selected_ids.add(a)
            trace.append({
                "source_id": a,
                "score": all_scores.get(a, 0.0),
                "expanded_via_relationship": True,
            })

    # --- Minimum diversity check ---
    # Ensure at least one source of each kind (policy / record) when both
    # kinds exist in the manifest.
    manifest_kinds: set[str] = set(all_kinds.values())
    selected_kinds = {all_kinds.get(sid, "") for sid in selected_ids}

    for required_kind in manifest_kinds:
        if required_kind and required_kind not in selected_kinds:
            # Pick the highest-scoring source of this kind not already selected
            best_id = None
            best_score = -1.0
            for sid, score in all_scores.items():
                if all_kinds.get(sid) == required_kind and sid not in selected_ids:
                    if score > best_score:
                        best_score = score
                        best_id = sid
            if best_id is not None:
                selected_ids.add(best_id)
                trace.append({
                    "source_id": best_id,
                    "score": best_score,
                    "expanded_via_relationship": False,
                })
                logger.info(
                    "Diversity: added %s (kind=%s) to pre-filter set",
                    best_id, required_kind,
                )

    # --- Fallback ---
    if len(selected_ids) < 3:
        logger.warning(
            "Pre-filter selected only %d sources — falling back to full manifest",
            len(selected_ids),
        )
        return get_manifest_index(), []

    # --- Build filtered manifest dict ---
    filtered = _filter_manifest_raw(raw, selected_ids)
    formatted = format_manifest_index(filtered)

    return formatted, trace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_manifest_raw(raw: dict, selected_ids: set[str]) -> dict:
    """Return a copy of *raw* containing only sources in *selected_ids*."""
    filtered: dict[str, Any] = {}

    if raw.get("domain_context"):
        filtered["domain_context"] = raw["domain_context"]

    for section in ("pdfs", "tables"):
        entries = [e for e in raw.get(section, []) if e["id"] in selected_ids]
        if entries:
            filtered[section] = copy.deepcopy(entries)

    # Keep only relationships where both sources are in the selected set
    rels = []
    for rel in raw.get("relationships", []):
        sources = rel.get("sources", [])
        if len(sources) == 2 and sources[0] in selected_ids and sources[1] in selected_ids:
            rels.append(copy.deepcopy(rel))
    if rels:
        filtered["relationships"] = rels

    return filtered


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.manifest_prefilter
# ---------------------------------------------------------------------------

def test_manifest_prefilter():
    import tempfile
    from unittest.mock import patch

    # Minimal manifest for testing
    test_manifest = {
        "pdfs": [
            {
                "id": "policy_a",
                "kind": "policy",
                "name": "Policy A",
                "summary": "Rules about travel expenses and reimbursement",
                "contains": ["expense limits", "receipt requirements"],
            },
            {
                "id": "policy_b",
                "kind": "policy",
                "name": "Policy B",
                "summary": "HR handbook covering leave and performance",
                "contains": ["leave entitlements", "performance reviews"],
            },
        ],
        "tables": [
            {
                "id": "employees",
                "kind": "record",
                "name": "Employees",
                "summary": "Employee roster with departments and hire dates",
                "contains": ["employee names", "departments", "hire dates"],
            },
            {
                "id": "projects",
                "kind": "record",
                "name": "Projects",
                "summary": "Active projects with budgets and deadlines",
                "contains": ["project names", "budgets", "deadlines"],
            },
        ],
        "relationships": [
            {
                "sources": ["employees", "projects"],
                "shared_key": "department",
                "description": "employees linked to projects by department",
                "verified": True,
            },
        ],
    }

    # Use a persistent tmpdir on Windows to avoid file-lock cleanup errors
    tmpdir = tempfile.mkdtemp()
    chroma_path = tmpdir

    # Patch the module-level names used by build_source_index / prefilter_manifest.
    # When run via `python -m`, the module is __main__, so we patch globals directly.
    import sys
    _this = sys.modules[__name__]
    with (
        patch.object(_this, "get_manifest_index_raw", return_value=test_manifest),
        patch.object(_this, "get_manifest_index", return_value="FULL MANIFEST FALLBACK"),
    ):
        global _source_index_stale
        _source_index_stale = True

        # Test build_source_index
        build_source_index(chroma_path=chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)
        col = client.get_collection(_COLLECTION_NAME)
        assert col.count() == 4, f"Expected 4 sources, got {col.count()}"

        # Test prefilter_manifest — query about expenses
        text, trace = prefilter_manifest(
            "What are the expense receipt requirements?",
            chroma_path=chroma_path,
        )
        trace_ids = {t["source_id"] for t in trace}
        assert "policy_a" in trace_ids, (
            f"Expected policy_a in trace, got {trace_ids}"
        )
        assert all("score" in t for t in trace), "Missing score in trace"
        assert "AVAILABLE DATA SOURCES" in text, "Filtered manifest missing header"

        # Test relationship expansion — query about projects should pull in employees
        text2, trace2 = prefilter_manifest(
            "What projects are in the engineering department?",
            chroma_path=chroma_path,
        )
        trace2_ids = {t["source_id"] for t in trace2}
        assert "projects" in trace2_ids, f"Expected projects, got {trace2_ids}"
        # employees should be pulled in via relationship
        assert "employees" in trace2_ids, (
            f"Expected employees via relationship expansion, got {trace2_ids}"
        )

        # Test diversity — a query that only matches records should still
        # include a policy source
        text3, trace3 = prefilter_manifest(
            "List all employee names and their departments",
            chroma_path=chroma_path,
        )
        trace3_kinds = set()
        for t in trace3:
            sid = t["source_id"]
            for section in ("pdfs", "tables"):
                for e in test_manifest.get(section, []):
                    if e["id"] == sid:
                        trace3_kinds.add(e["kind"])
        assert "policy" in trace3_kinds, (
            f"Diversity check failed — no policy source in {trace3}"
        )

        # Test stale flag triggers rebuild
        _source_index_stale = True
        text4, trace4 = prefilter_manifest(
            "travel expenses",
            chroma_path=chroma_path,
        )
        assert not _source_index_stale, "Stale flag should be cleared after rebuild"
        assert len(trace4) > 0, "Should have trace entries after rebuild"

        print("All tests passed.")


if __name__ == "__main__":
    test_manifest_prefilter()
