"""
core/data_context.py — Domain summary derived at runtime from the manifest.

The Chat agent's reasoning call needs to know, at a high level, what the
system can answer — enough to route out-of-scope queries (e.g. "what's the
weather?") to a DIRECT response without pretending the pipeline can handle
them. It must NOT be a hardcoded list of source names; if the deployment
changes its data, the context changes automatically.

Public:
  get_data_context() -> str

Cached in a module-level dict. Invalidated automatically when the manifest
cache is cleared (register_invalidation_callback).
"""

from __future__ import annotations

import logging

from core.manifest import get_manifest_index_raw, register_invalidation_callback

logger = logging.getLogger(__name__)


_cache: dict[str, str] = {}


def _invalidate() -> None:
    _cache.clear()


register_invalidation_callback(_invalidate)


def _build_summary(raw: dict) -> str:
    lines: list[str] = ["The system contains the following data:"]

    def _fmt_entry(entry: dict) -> str | None:
        name = entry.get("name") or entry.get("id")
        if not name:
            return None
        summary = (entry.get("summary") or "").strip()
        kind = (entry.get("kind") or "").strip()
        if kind and summary:
            return f"- {name} ({kind}): {summary}"
        if kind:
            return f"- {name} ({kind})"
        if summary:
            return f"- {name}: {summary}"
        return f"- {name}"

    has_any = False
    for entry in raw.get("pdfs", []) + raw.get("tables", []):
        line = _fmt_entry(entry)
        if line:
            lines.append(line)
            has_any = True

    if not has_any:
        return "The system has no ingested data yet."

    domain_context = (raw.get("domain_context") or "").strip()
    if domain_context:
        lines.append("")
        lines.append(f"Domain context: {domain_context}")

    return "\n".join(lines)


def get_data_context() -> str:
    """Return a short bullet summary of what the system's data covers.

    Derived from the manifest index. Cached until the manifest cache is
    invalidated. Domain-agnostic — never hardcodes source names.
    """
    if "summary" not in _cache:
        raw = get_manifest_index_raw()
        _cache["summary"] = _build_summary(raw)
    return _cache["summary"]


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m core.data_context
# ---------------------------------------------------------------------------

def test_data_context():
    from unittest.mock import patch

    # Patch the imported alias in this module's own namespace so the test works
    # whether imported as core.data_context or run via python -m core.data_context.
    patch_target = f"{__name__}.get_manifest_index_raw"

    # 1. Well-formed manifest with both pdfs and tables, kinds present
    fake_raw_full = {
        "domain_context": "records describe employees; policies describe rules",
        "pdfs": [
            {"id": "policy_a", "name": "Policy A", "summary": "rules for X", "kind": "policy"},
            {"id": "policy_b", "name": "Policy B", "summary": "rules for Y"},
        ],
        "tables": [
            {"id": "records_a", "name": "Records A", "summary": "entity list", "kind": "record"},
            {"id": "records_b", "name": "Records B"},
        ],
    }
    _cache.clear()
    with patch(patch_target, return_value=fake_raw_full):
        out = get_data_context()

    assert "Policy A (policy): rules for X" in out
    assert "Policy B: rules for Y"            in out
    assert "Records A (record): entity list"  in out
    assert "Records B"                         in out
    assert "Domain context:" in out
    print("PASS: full manifest renders name/summary/kind correctly")

    # 2. Cache hit — next call returns same string without consulting manifest
    with patch(patch_target,
               side_effect=AssertionError("manifest should not be re-read on cache hit")):
        cached = get_data_context()
    assert cached == out
    print("PASS: cached on second call")

    # 3. Invalidation clears cache
    _invalidate()
    assert _cache == {}
    print("PASS: invalidation clears cache")

    # 4. Empty manifest → explicit fallback
    _cache.clear()
    with patch(patch_target, return_value={"pdfs": [], "tables": []}):
        out = get_data_context()
    assert "no ingested data" in out.lower()
    print("PASS: empty manifest yields fallback message")

    # 5. Entries missing summary/kind are still rendered by name
    _cache.clear()
    with patch(patch_target, return_value={
        "pdfs":   [{"id": "p1", "name": "Only Name"}],
        "tables": [],
    }):
        out = get_data_context()
    assert "- Only Name" in out
    print("PASS: bare entries render by name only")

    _cache.clear()
    print("PASS: all data_context tests passed")


if __name__ == "__main__":
    test_data_context()
