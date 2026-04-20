"""
core/data_context.py — Read the data_context paragraph from the manifest index.

The paragraph itself is produced at ingest time by
ingestion/manifest_writer.py::regenerate_data_context() and stored in the
top-level `data_context` field of manifest_index.yaml. This module is a
thin synchronous reader — no LLM calls, no cache of its own (relies on
core/manifest.py's manifest cache).

Public:
  get_data_context() -> str
"""

from __future__ import annotations

import logging

from core.manifest import get_manifest_index_raw

logger = logging.getLogger(__name__)


_GENERIC_FALLBACK = "The system answers questions from the currently ingested data sources."


def get_data_context() -> str:
    """Return the LLM-generated data context paragraph from the manifest.

    The paragraph is written at ingest time. Returns a generic fallback
    string when the field is missing — e.g. on a freshly cloned checkout
    whose manifest predates this feature.
    """
    raw = get_manifest_index_raw()
    text = (raw.get("data_context") or "").strip()
    return text or _GENERIC_FALLBACK


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.data_context
# ---------------------------------------------------------------------------

def test_data_context():
    from unittest.mock import patch

    patch_target = f"{__name__}.get_manifest_index_raw"

    # 1. data_context field present → returned verbatim
    paragraph = (
        "This system helps you look up company policies and employee records, "
        "including rules about day-to-day operations and individual details."
    )
    with patch(patch_target, return_value={
        "pdfs":         [{"id": "p1", "name": "Some Policy"}],
        "tables":       [{"id": "t1", "name": "Some Table"}],
        "data_context": paragraph,
    }):
        out = get_data_context()
    assert out == paragraph
    print("PASS: data_context field is returned verbatim")

    # 2. data_context field missing → generic fallback
    with patch(patch_target, return_value={
        "pdfs":   [{"id": "p1", "name": "X"}],
        "tables": [],
    }):
        out = get_data_context()
    assert out == _GENERIC_FALLBACK
    print("PASS: missing data_context field yields generic fallback")

    # 3. data_context field empty string → generic fallback
    with patch(patch_target, return_value={
        "pdfs":         [{"id": "p1", "name": "X"}],
        "tables":       [],
        "data_context": "   ",
    }):
        out = get_data_context()
    assert out == _GENERIC_FALLBACK
    print("PASS: empty/whitespace data_context yields generic fallback")

    # 4. Manifest empty (no pdfs/tables) and no data_context → generic fallback
    with patch(patch_target, return_value={"pdfs": [], "tables": []}):
        out = get_data_context()
    assert out == _GENERIC_FALLBACK
    print("PASS: empty manifest without data_context yields generic fallback")

    # 5. Sentinel check — reader does not synthesize; it just returns the field.
    # If the paragraph happens to reference a sentinel, the reader passes it
    # through. The no-source-name contract is owned by the ingestion writer,
    # not this reader. This test pins that contract: get_data_context returns
    # whatever text the writer put in the manifest.
    sentinel_paragraph = "The system contains ZZZ_SENTINEL (this would be a bug in the writer)."
    with patch(patch_target, return_value={
        "pdfs":         [{"id": "ZZZ_SENTINEL"}],
        "tables":       [],
        "data_context": sentinel_paragraph,
    }):
        out = get_data_context()
    assert out == sentinel_paragraph
    print("PASS: get_data_context is a pass-through; source-name contract is owned by the writer")

    print("PASS: all data_context tests passed")


if __name__ == "__main__":
    test_data_context()
