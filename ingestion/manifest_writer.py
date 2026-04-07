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

Both functions determine which section (pdfs | tables) to write into by
inspecting the detail_entry["type"] field:
    type == "pdf"           → pdfs section
    type == "csv" | "sqlite" → tables section
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.manifest import invalidate_manifest_cache

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    idx_data[section], _ = _upsert_entry(idx_data[section], source_id, index_entry)
    _write_yaml(idx_path, idx_data)

    # ── manifest_detail.yaml ──────────────────────────────────────
    det_data = _read_yaml(det_path)
    det_data.setdefault("pdfs",   [])
    det_data.setdefault("tables", [])
    det_data[section], _ = _upsert_entry(det_data[section], source_id, detail_entry)
    _write_yaml(det_path, det_data)

    invalidate_manifest_cache()


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

    print("\nPASS: all manifest_writer tests passed")


if __name__ == "__main__":
    test_manifest_writer()
