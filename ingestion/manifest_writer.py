"""
ingestion/manifest_writer.py — Write and delete entries in both manifest files.

Public API
----------
write_source_to_manifest(source_id, index_entry, detail_entry)
    Append or overwrite one source in manifest_index.yaml and
    manifest_detail.yaml. If source_id already exists, replace it in-place.
    If not, append it. Calls invalidate_manifest_cache() after writing.

delete_source_from_manifest(source_id)
    Remove one source from both manifest files.
    Raises ValueError if source_id is not found in either file.
    Calls invalidate_manifest_cache() after writing.

regenerate_data_context()
    Produce the top-level `data_context` paragraph for manifest_index.yaml
    via a single LLM call. Domain-agnostic — never names specific sources.
    Called at ingest time so query-time readers can load it synchronously.

Both source-write functions determine which section (pdfs | tables) to
write into by inspecting the detail_entry["type"] field:
    type == "pdf"           → pdfs section
    type == "csv" | "sqlite" → tables section
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from core.manifest import invalidate_manifest_cache

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


_EMPTY_FALLBACK   = "The system has no ingested data yet."
_GENERIC_FALLBACK = "The system answers questions from the currently ingested data sources."


_DATA_CONTEXT_SYSTEM_PROMPT = """\
You write a one-paragraph user-facing description of what a data-answering
system can help with, based on a manifest of its ingested data. Your output
feeds a chat agent's out-of-scope detection: it tells the agent what kinds
of questions fall inside the system's knowledge.

Requirements:
  - Produce ONE paragraph, 2 to 4 sentences, in plain natural language.
  - Describe the DOMAINS, TOPICS, and KINDS OF QUESTIONS the system can
    answer. Speak in general terms a non-technical reader would understand.
  - Do NOT name any specific source, file, table, column, entity, or
    identifier from the manifest — even if the manifest lists them plainly.
  - Do NOT mention file types (PDF, CSV, SQLite, spreadsheet, database,
    document, table).
  - Do NOT enumerate or list items. Do not use bullets.
  - Do NOT say "the manifest", "the index", "this system contains N
    sources", or anything that references the underlying structure.
  - If the manifest is sparse or very narrow, describe what is there in
    general terms rather than inventing scope.

Output ONLY the paragraph — no preamble, no JSON, no headings.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    """Read and return a YAML file as a dict. Returns empty dict on missing file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict) -> None:
    """Write a dict to a YAML file, preserving the leading comment block."""
    # Preserve any leading comment lines from the existing file
    header_lines: list[str] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    break

    with open(path, "w", encoding="utf-8") as f:
        if header_lines:
            f.writelines(header_lines)
            f.write("\n")
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _section_for(detail_entry: dict) -> str:
    """
    Return "pdfs" or "tables" based on detail_entry["type"].

    Raises ValueError for unrecognised types.
    """
    source_type = str(detail_entry.get("type", "")).lower()
    if source_type == "pdf":
        return "pdfs"
    if source_type in ("csv", "sqlite"):
        return "tables"
    raise ValueError(
        f"Cannot determine manifest section: detail_entry['type'] is "
        f"{source_type!r}. Expected 'pdf', 'csv', or 'sqlite'."
    )


def _upsert_entry(entries: list[dict], source_id: str, new_entry: dict) -> tuple[list[dict], bool]:
    """
    Replace the entry with matching id, or append if not found.

    Returns (updated_list, was_replaced).
    """
    for i, entry in enumerate(entries):
        if entry.get("id") == source_id:
            entries[i] = new_entry
            return entries, True
    entries.append(new_entry)
    return entries, False


def _remove_entry(entries: list[dict], source_id: str) -> tuple[list[dict], bool]:
    """
    Remove the entry with matching id.

    Returns (updated_list, was_found).
    """
    original_len = len(entries)
    filtered = [e for e in entries if e.get("id") != source_id]
    return filtered, len(filtered) < original_len


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_source_to_manifest(
    source_id: str,
    index_entry: dict[str, Any],
    detail_entry: dict[str, Any],
    *,
    index_path: Path | None = None,
    detail_path: Path | None = None,
) -> None:
    """
    Append or overwrite one source in both manifest YAML files.

    If source_id already exists in a section, that entry is replaced in-place.
    If not, the entry is appended to the appropriate section.

    Parameters
    ----------
    source_id   : The id string used in both manifest files.
    index_entry : Full dict for manifest_index.yaml (must include 'id').
    detail_entry: Full dict for manifest_detail.yaml (must include 'id' and 'type').
    index_path  : Override path to manifest_index.yaml (used in tests).
    detail_path : Override path to manifest_detail.yaml (used in tests).
    """
    from core.llm_config import _load_config
    paths = _load_config()["paths"]

    idx_path = index_path  or (_PROJECT_ROOT / paths["manifest_index"])
    det_path = detail_path or (_PROJECT_ROOT / paths["manifest_detail"])

    section = _section_for(detail_entry)

    # ── manifest_index.yaml ───────────────────────────────────────
    idx_data = _read_yaml(idx_path)
    idx_data.setdefault("pdfs",   [])
    idx_data.setdefault("tables", [])

    # Preserve human-authored fields from existing entry
    for existing in idx_data[section]:
        if existing.get("id") == source_id:
            if existing.get("notes") and not index_entry.get("notes"):
                index_entry["notes"] = existing["notes"]
            break

    # Preserve top-level domain_context
    if idx_data.get("domain_context") and "domain_context" not in idx_data:
        pass  # nothing to do
    # (domain_context lives at top level; we just don't overwrite it)

    idx_data[section], _ = _upsert_entry(idx_data[section], source_id, index_entry)
    _write_yaml(idx_path, idx_data)

    # ── manifest_detail.yaml ──────────────────────────────────────
    det_data = _read_yaml(det_path)
    det_data.setdefault("pdfs",   [])
    det_data.setdefault("tables", [])

    # Preserve human-authored fields from existing entry
    for existing in det_data[section]:
        if existing.get("id") == source_id:
            if existing.get("notes") and not detail_entry.get("notes"):
                detail_entry["notes"] = existing["notes"]
            break

    det_data[section], _ = _upsert_entry(det_data[section], source_id, detail_entry)
    _write_yaml(det_path, det_data)

    invalidate_manifest_cache()


def write_cross_source_relationships(
    relationships: list[dict[str, Any]],
    *,
    index_path: Path | None = None,
) -> None:
    """
    Write cross-source relationships to manifest_index.yaml.

    Preserves existing domain_context and all source entries.

    Parameters
    ----------
    relationships : List of {sources, shared_key, description, verified} dicts.
    index_path    : Override path (used in tests).
    """
    from core.llm_config import _load_config
    paths = _load_config()["paths"]
    idx_path = index_path or (_PROJECT_ROOT / paths["manifest_index"])

    idx_data = _read_yaml(idx_path)
    idx_data["relationships"] = relationships
    _write_yaml(idx_path, idx_data)
    invalidate_manifest_cache()


def update_table_relationships(
    source_id: str,
    relationships: list[dict[str, Any]],
    *,
    detail_path: Path | None = None,
) -> None:
    """
    Update the relationships field for a single table in manifest_detail.yaml.

    Parameters
    ----------
    source_id     : The table's id in manifest_detail.yaml.
    relationships : List of {from_column, to_table, to_column, verified} dicts.
    detail_path   : Override path (used in tests).

    Raises
    ------
    ValueError
        If source_id is not found in the tables section.
    """
    from core.llm_config import _load_config
    paths = _load_config()["paths"]
    det_path = detail_path or (_PROJECT_ROOT / paths["manifest_detail"])

    det_data = _read_yaml(det_path)
    for entry in det_data.get("tables", []):
        if entry["id"] == source_id:
            entry["relationships"] = relationships
            _write_yaml(det_path, det_data)
            invalidate_manifest_cache()
            return

    raise ValueError(
        f"source_id '{source_id}' not found in tables section of manifest_detail.yaml"
    )


def delete_source_from_manifest(
    source_id: str,
    *,
    index_path: Path | None = None,
    detail_path: Path | None = None,
) -> None:
    """
    Remove one source from both manifest YAML files.

    Parameters
    ----------
    source_id   : The id string to remove.
    index_path  : Override path to manifest_index.yaml (used in tests).
    detail_path : Override path to manifest_detail.yaml (used in tests).

    Raises
    ------
    ValueError
        If source_id is not found in either manifest file.
    """
    from core.llm_config import _load_config
    paths = _load_config()["paths"]

    idx_path = index_path  or (_PROJECT_ROOT / paths["manifest_index"])
    det_path = detail_path or (_PROJECT_ROOT / paths["manifest_detail"])

    idx_data = _read_yaml(idx_path)
    det_data = _read_yaml(det_path)

    found_in_index  = False
    found_in_detail = False

    for section in ("pdfs", "tables"):
        updated, found = _remove_entry(idx_data.get(section, []), source_id)
        if found:
            idx_data[section] = updated
            found_in_index = True

        updated, found = _remove_entry(det_data.get(section, []), source_id)
        if found:
            det_data[section] = updated
            found_in_detail = True

    if not found_in_index and not found_in_detail:
        raise ValueError(
            f"source_id '{source_id}' not found in either manifest file. "
            f"Cannot delete a source that does not exist."
        )

    _write_yaml(idx_path, idx_data)
    _write_yaml(det_path, det_data)

    invalidate_manifest_cache()


async def regenerate_data_context(
    *,
    index_path: Path | None = None,
) -> str:
    """
    Produce the top-level `data_context` paragraph for manifest_index.yaml.

    Reads the current manifest index, asks the ingestion LLM for a short
    natural-language paragraph describing the domains, topics, and kinds
    of questions this system can answer, and writes the paragraph back
    under the top-level `data_context` key. Invalidates the manifest
    cache on write.

    The prompt is domain-agnostic — the LLM is instructed never to name
    any source, table, column, entity, or file type. Intended to be
    called from ingestion entry points (per-source upload and end of
    batch), not from the query hot path.

    Returns the paragraph that was written:
      - The LLM output on success.
      - `_EMPTY_FALLBACK` when the manifest has no sources.
      - `_GENERIC_FALLBACK` when the LLM call fails or returns empty.
    """
    from core.llm_config import _load_config, get_llm
    if index_path is None:
        paths = _load_config()["paths"]
        idx_path = _PROJECT_ROOT / paths["manifest_index"]
    else:
        idx_path = index_path

    idx_data = _read_yaml(idx_path)
    pdfs   = idx_data.get("pdfs")   or []
    tables = idx_data.get("tables") or []

    if not pdfs and not tables:
        paragraph = _EMPTY_FALLBACK
        idx_data["data_context"] = paragraph
        _write_yaml(idx_path, idx_data)
        invalidate_manifest_cache()
        return paragraph

    manifest_view = {
        "pdfs":             pdfs,
        "tables":           tables,
        "relationships":    idx_data.get("relationships")    or [],
        "domain_context":   idx_data.get("domain_context")   or "",
    }
    user_message = (
        "MANIFEST INDEX:\n"
        f"{json.dumps(manifest_view, indent=2, ensure_ascii=False)}\n\n"
        "Produce the paragraph."
    )

    try:
        llm = get_llm("planner")
        resp = await llm.ainvoke([
            {"role": "system", "content": _DATA_CONTEXT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ])
        paragraph = (resp.content or "").strip()
        if not paragraph:
            raise ValueError("LLM returned empty response")
    except Exception as exc:
        logger.warning(
            "regenerate_data_context: LLM call failed (%s); using generic fallback.",
            exc,
        )
        paragraph = _GENERIC_FALLBACK

    idx_data["data_context"] = paragraph
    _write_yaml(idx_path, idx_data)
    invalidate_manifest_cache()
    return paragraph


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m ingestion.manifest_writer
# ---------------------------------------------------------------------------

def test_manifest_writer():
    import tempfile
    import os
    from unittest.mock import patch

    # ── Fixture entries ───────────────────────────────────────────
    CSV_INDEX = {
        "id":      "vendor_contacts",
        "name":    "Vendor Contacts",
        "summary": "List of approved vendors with contact details.",
    }
    CSV_DETAIL = {
        "id":       "vendor_contacts",
        "filename": "vendor_contacts.csv",
        "type":     "csv",
        "row_count_approx": 120,
        "columns": [
            {"name": "vendor_id",   "type": "integer", "description": "Unique vendor ID"},
            {"name": "vendor_name", "type": "string",  "description": "Vendor legal name"},
        ],
    }

    PDF_INDEX = {
        "id":      "expense_policy_2025",
        "name":    "Expense Policy 2025",
        "summary": "Updated reimbursement rules for FY2025.",
    }
    PDF_DETAIL = {
        "id":       "expense_policy_2025",
        "filename": "expense_policy_2025.pdf",
        "type":     "pdf",
        "pages":    12,
        "sections": ["Section 1: Meal Limits", "Section 2: Transport Caps"],
        "tags":     ["expenses", "reimbursement"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = Path(tmpdir) / "manifest_index.yaml"
        det_path = Path(tmpdir) / "manifest_detail.yaml"

        # Pre-populate the files with one existing entry each so we can
        # test both the append path and the overwrite path.
        existing_idx = {
            "pdfs":   [{"id": "travel_policy_2024", "name": "Travel Policy 2024",
                        "summary": "Existing PDF entry."}],
            "tables": [],
        }
        existing_det = {
            "pdfs":   [{"id": "travel_policy_2024", "filename": "travel_policy.pdf",
                        "type": "pdf", "pages": 18, "sections": []}],
            "tables": [],
        }
        _write_yaml(idx_path, existing_idx)
        _write_yaml(det_path, existing_det)

        # ── Test 1: write_source_to_manifest — append a new CSV ───
        patch_target = f"{__name__}.invalidate_manifest_cache"
        with patch(patch_target) as mock_inv:
            write_source_to_manifest(
                "vendor_contacts", CSV_INDEX, CSV_DETAIL,
                index_path=idx_path, detail_path=det_path,
            )
            mock_inv.assert_called_once()
        print("PASS: write_source_to_manifest called invalidate_manifest_cache()")

        idx_after = _read_yaml(idx_path)
        det_after = _read_yaml(det_path)

        assert any(e["id"] == "vendor_contacts" for e in idx_after["tables"]), \
            "New CSV entry should appear in tables section of index"
        assert any(e["id"] == "vendor_contacts" for e in det_after["tables"]), \
            "New CSV entry should appear in tables section of detail"
        # Original PDF entry must still be present
        assert any(e["id"] == "travel_policy_2024" for e in idx_after["pdfs"]), \
            "Existing PDF entry must not be removed on append"
        print("PASS: new CSV source appended; existing PDF source preserved")

        # ── Test 2: write_source_to_manifest — append a new PDF ───
        write_source_to_manifest(
            "expense_policy_2025", PDF_INDEX, PDF_DETAIL,
            index_path=idx_path, detail_path=det_path,
        )
        idx_after2 = _read_yaml(idx_path)
        det_after2 = _read_yaml(det_path)

        assert any(e["id"] == "expense_policy_2025" for e in idx_after2["pdfs"]), \
            "New PDF entry should appear in pdfs section of index"
        assert any(e["id"] == "expense_policy_2025" for e in det_after2["pdfs"]), \
            "New PDF entry should appear in pdfs section of detail"
        assert len(idx_after2["pdfs"]) == 2, \
            "pdfs section should now have two entries"
        print("PASS: new PDF source appended to pdfs section")

        # ── Test 3: write_source_to_manifest — overwrite existing ─
        updated_csv_index = {**CSV_INDEX, "summary": "UPDATED summary."}
        updated_csv_detail = {**CSV_DETAIL, "row_count_approx": 999}

        write_source_to_manifest(
            "vendor_contacts", updated_csv_index, updated_csv_detail,
            index_path=idx_path, detail_path=det_path,
        )
        idx_after3 = _read_yaml(idx_path)
        det_after3 = _read_yaml(det_path)

        tables_idx = idx_after3["tables"]
        tables_det = det_after3["tables"]
        assert len(tables_idx) == 1, "Overwrite must not duplicate the entry"
        assert tables_idx[0]["summary"] == "UPDATED summary.", \
            "Index entry summary must be updated"
        assert tables_det[0]["row_count_approx"] == 999, \
            "Detail entry row_count_approx must be updated"
        print("PASS: overwrite replaces entry in-place without duplication")

        # ── Test 4: delete_source_from_manifest — remove existing ─
        with patch(patch_target) as mock_inv:
            delete_source_from_manifest(
                "vendor_contacts",
                index_path=idx_path, detail_path=det_path,
            )
            mock_inv.assert_called_once()
        print("PASS: delete_source_from_manifest called invalidate_manifest_cache()")

        idx_after4 = _read_yaml(idx_path)
        det_after4 = _read_yaml(det_path)

        assert not any(e["id"] == "vendor_contacts" for e in idx_after4.get("tables", [])), \
            "Deleted source must not appear in index tables section"
        assert not any(e["id"] == "vendor_contacts" for e in det_after4.get("tables", [])), \
            "Deleted source must not appear in detail tables section"
        assert any(e["id"] == "travel_policy_2024" for e in idx_after4.get("pdfs", [])), \
            "Unrelated PDF entry must not be affected by deletion"
        print("PASS: source deleted from both files; unrelated entries preserved")

        # ── Test 5: delete_source_from_manifest — unknown id raises ─
        try:
            delete_source_from_manifest(
                "nonexistent_source_xyz",
                index_path=idx_path, detail_path=det_path,
            )
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "nonexistent_source_xyz" in str(exc)
            print(f"PASS: ValueError raised for unknown source_id: {exc}")

        # ── Test 6: _section_for raises on unknown type ────────────
        try:
            _section_for({"type": "excel"})
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "excel" in str(exc)
            print(f"PASS: _section_for raises ValueError for unknown type: {exc}")

        # ── Test 7: regenerate_data_context — empty manifest, no LLM ─
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        empty_idx_path = Path(tmpdir) / "empty_index.yaml"
        _write_yaml(empty_idx_path, {"pdfs": [], "tables": []})

        llm_patch = "core.llm_config.get_llm"
        no_llm = MagicMock()
        with patch(llm_patch, return_value=no_llm):
            paragraph = asyncio.run(regenerate_data_context(index_path=empty_idx_path))
        assert paragraph == _EMPTY_FALLBACK
        no_llm.ainvoke.assert_not_called() if hasattr(no_llm, "ainvoke") else None
        reloaded = _read_yaml(empty_idx_path)
        assert reloaded.get("data_context") == _EMPTY_FALLBACK
        print("PASS: regenerate_data_context writes _EMPTY_FALLBACK on empty manifest without calling LLM")

        # ── Test 8: regenerate_data_context — populated manifest, LLM called, sentinel not leaked ──
        sentinel = "ZZZ_SENTINEL_SOURCE"
        populated_idx_path = Path(tmpdir) / "populated_index.yaml"
        _write_yaml(populated_idx_path, {
            "pdfs":   [{"id": sentinel, "name": sentinel, "summary": "some policy",
                        "contains": ["some topic"]}],
            "tables": [],
        })

        generic_paragraph = (
            "This system answers questions about corporate policies and employee "
            "records, including rules governing day-to-day operations."
        )
        fake_resp = MagicMock(); fake_resp.content = generic_paragraph
        fake_llm = MagicMock(); fake_llm.ainvoke = AsyncMock(return_value=fake_resp)
        with patch(llm_patch, return_value=fake_llm):
            paragraph2 = asyncio.run(regenerate_data_context(index_path=populated_idx_path))

        fake_llm.ainvoke.assert_called_once()
        call_messages = fake_llm.ainvoke.call_args[0][0]
        # The LLM message is what we sent; the sentinel IS present in the input
        # (that's the whole manifest) — but the returned paragraph must not contain
        # it, and the written data_context must match the LLM output.
        user_msg = next(m["content"] for m in call_messages if m["role"] == "user")
        assert sentinel in user_msg, "manifest content should reach the LLM prompt"
        assert sentinel not in paragraph2, "returned paragraph must not contain the sentinel"
        reloaded2 = _read_yaml(populated_idx_path)
        assert reloaded2["data_context"] == generic_paragraph
        # Original source entry preserved
        assert reloaded2["pdfs"][0]["id"] == sentinel
        print("PASS: regenerate_data_context writes LLM paragraph, preserves sources, keeps sentinel out of output")

        # ── Test 9: regenerate_data_context — LLM exception → generic fallback ──
        err_llm = MagicMock(); err_llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        with patch(llm_patch, return_value=err_llm):
            paragraph3 = asyncio.run(regenerate_data_context(index_path=populated_idx_path))
        assert paragraph3 == _GENERIC_FALLBACK
        reloaded3 = _read_yaml(populated_idx_path)
        assert reloaded3["data_context"] == _GENERIC_FALLBACK
        print("PASS: regenerate_data_context falls back to generic paragraph on LLM error")

        # ── Test 10: system prompt is domain-agnostic (no hardcoded source names) ──
        prompt_lower = _DATA_CONTEXT_SYSTEM_PROMPT.lower()
        for banned in ("employee", "travel", "policy", "salary", "clearance",
                       "department", "vendor", "noa", "dan"):
            assert banned not in prompt_lower, \
                f"Domain-specific token {banned!r} leaked into _DATA_CONTEXT_SYSTEM_PROMPT"
        print("PASS: _DATA_CONTEXT_SYSTEM_PROMPT contains no domain-specific tokens")

    print("\nPASS: all manifest_writer tests passed")


if __name__ == "__main__":
    test_manifest_writer()
