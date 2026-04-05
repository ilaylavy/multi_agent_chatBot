"""
agents/data_scientist.py — Data Scientist worker (structured data queries).

This is a registry worker callable, NOT a LangGraph node.
Signature: async data_scientist_worker(state, task) -> TaskResult

Flow:
  manifest detail (type: csv|sqlite) → LLM generates query →
  sandboxed execution against data/tables/ → TaskResult

Rules enforced here:
  - Never guess a number: if the table file is missing, return success=False.
  - Never let a bad LLM query crash the system: all execution is sandboxed.
  - The manifest type field determines pandas vs SQL — no hardcoding.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from core.llm_config import _load_config, get_llm
from core.manifest import _load_detail_raw, get_manifest_detail
from core.parse import parse_llm_json
from core.state import AgentState, Task, TaskResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tables_dir() -> Path:
    """Return the tables directory path from config.yaml."""
    return _PROJECT_ROOT / _load_config()["paths"]["tables"]


def _get_raw_entry(source_id: str) -> dict:
    """
    Return the raw manifest_detail.yaml entry for a given source_id.
    Used to extract filename and type without re-reading the YAML.
    """
    raw = _load_detail_raw()
    for section in ("pdfs", "tables"):
        for entry in raw.get(section, []):
            if entry["id"] == source_id:
                return entry
    raise ValueError(f"source_id '{source_id}' not found in manifest_detail.yaml")


def _execute_pandas(query: str, file_path: Path) -> tuple[Any, int]:
    """
    Load CSV into a DataFrame and eval the LLM-generated query expression.

    The query must reference `df` — e.g. `df[df['col'] == 'val']`.
    Sandboxed: __builtins__ is empty so import/exec/os are not accessible.

    Returns (result, row_count).
    """
    df = pd.read_csv(file_path)
    safe_globals = {"__builtins__": {}, "pd": pd, "df": df}
    result = eval(query, safe_globals)  # noqa: S307 — intentionally sandboxed
    if isinstance(result, pd.DataFrame):
        row_count = len(result)
        result_value = result.to_dict(orient="records")
    elif isinstance(result, pd.Series):
        row_count = len(result)
        result_value = result.tolist()
    else:
        row_count = 1
        result_value = result
    return result_value, row_count


def _execute_sql(query: str, file_path: Path) -> tuple[Any, int]:
    """
    Execute a SQL SELECT query against the SQLite database at file_path.

    Only SELECT statements are permitted. Any other statement raises ValueError.
    Returns (rows_as_list_of_dicts, row_count).
    """
    stripped = query.strip().upper()
    if not stripped.startswith("SELECT"):
        raise ValueError(
            f"Only SELECT statements are permitted. Got: {query[:80]}"
        )
    conn = sqlite3.connect(str(file_path))
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return rows, len(rows)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a data query specialist. You will be given a task and the full schema
of a data table. Your job is to write a precise query that retrieves exactly
what the task requires.

Rules:
  - For CSV tables  : write a pandas expression using `df` as the DataFrame variable.
                      The expression must evaluate to a result (scalar, list, or DataFrame).
                      Do NOT write `df = pd.read_csv(...)` — df is already loaded.
  - For SQLite tables: write a standard SQL SELECT statement.
  - Never guess or invent values — only query what is in the table.
  - If the task cannot be answered from this table, set query to an empty string
    and explain why in the explanation field.

Respond with ONLY a JSON object matching this schema — no explanation, no markdown:
{
  "query_type":  "pandas" or "sql",
  "query":       "the exact query string to execute",
  "explanation": "one sentence describing what this query retrieves"
}
"""

_USER_TEMPLATE = """\
TABLE SCHEMA:
{manifest_detail}

TASK:
{task_description}
"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def data_scientist_view(state: AgentState, task: Task, manifest_detail: str) -> dict:
    """
    Returns only the assigned task and manifest detail for that table.
    The LLM prompt is built from this view — nothing else from state is visible.
    """
    return {
        "task":            task,
        "manifest_detail": manifest_detail,
    }


# ---------------------------------------------------------------------------
# Worker callable
# ---------------------------------------------------------------------------

async def data_scientist_worker(
    state: AgentState,
    task: Task,
) -> TaskResult:
    """
    Registry worker callable — dispatched by the Router via asyncio.gather.

    Parameters
    ----------
    state : Full AgentState — filtered by data_scientist_view before use.
    task  : The single Task assigned to this worker.
    """
    source_id = task["source_id"]

    # ── Manifest ──────────────────────────────────────────────────
    manifest_detail = get_manifest_detail(source_id)
    raw_entry       = _get_raw_entry(source_id)
    source_type     = raw_entry.get("type", "").lower()   # "csv" | "sqlite"
    filename        = raw_entry.get("filename", "")
    table_name      = raw_entry.get("table_name", filename)

    view = data_scientist_view(state, task, manifest_detail)

    # ── File existence check — never guess numbers ────────────────
    file_path = _tables_dir() / filename
    if not file_path.exists():
        return TaskResult(
            task_id=task["task_id"],
            worker_type="data_scientist",
            output=json.dumps({"error": f"Table file not found: {file_path}"}),
            success=False,
            error=f"Table file not found: {file_path}. "
                  f"Place the file in {_tables_dir()} and retry.",
        )

    # ── LLM query generation ──────────────────────────────────────
    user_message = _USER_TEMPLATE.format(
        manifest_detail=view["manifest_detail"],
        task_description=view["task"]["description"],
    )

    llm = get_llm("data_scientist")
    response = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    # ── Parse LLM response — per CLAUDE.md LLM Output Parsing rule ──
    data = parse_llm_json(response.content)
    try:
        query_type = data["query_type"]
        query      = data["query"]
    except KeyError as exc:
        raise ValueError(
            f"Missing key in LLM output: {exc}\nRaw output: {response.content}"
        ) from exc

    # ── Sandboxed execution ───────────────────────────────────────
    try:
        if query_type == "pandas":
            result_value, row_count = _execute_pandas(query, file_path)
        elif query_type == "sql":
            result_value, row_count = _execute_sql(query, file_path)
        else:
            raise ValueError(f"Unknown query_type '{query_type}' — must be 'pandas' or 'sql'")
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for sandboxing
        return TaskResult(
            task_id=task["task_id"],
            worker_type="data_scientist",
            output=json.dumps({"error": str(exc), "query_used": query}),
            success=False,
            error=f"Query execution failed: {exc}",
        )

    output = json.dumps({
        "result_value": result_value,
        "query_used":   query,
        "table_name":   table_name,
        "row_count":    row_count,
    })

    return TaskResult(
        task_id=task["task_id"],
        worker_type="data_scientist",
        output=output,
        success=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.data_scientist
# ---------------------------------------------------------------------------

def test_data_scientist():
    import sys
    import os
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(_PROJECT_ROOT))

    fake_task: Task = {
        "task_id":     "t1",
        "worker_type": "data_scientist",
        "description": "Find the full name and clearance level of all level-A employees",
        "source_id":   "employees",
    }

    fake_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-session-001",
        "conversation_history": [],
        "plan":                 [fake_task],
        "manifest_context":     "",
        "task_results":         {},
        "sources_used":         [],
        "retrieved_chunks":     [],
        "draft_answer":         "",
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "retry_count":          0,
        "retry_notes":          "",
        "final_answer":         "",
        "final_sources":        [],
    }

    # ── Create a real fixture CSV in data/tables/ ─────────────────
    # When run under pytest the session-scoped fixture in conftest.py
    # creates this file before any test runs. When run standalone
    # (python -m agents.data_scientist) we create it here and clean up.
    tables_dir = _tables_dir()
    tables_dir.mkdir(parents=True, exist_ok=True)
    fixture_csv = tables_dir / "employees.csv"

    csv_content = (
        "employee_id,full_name,email,department,clearance_level,hire_date,manager_id\n"
        "1,Noa Levi,noa@corp.com,Engineering,A,2020-01-15,\n"
        "2,Dan Cohen,dan@corp.com,Finance,B,2019-06-01,1\n"
        "3,Yael Ben,yael@corp.com,HR,A,2021-03-20,1\n"
    )
    _created_fixture = not fixture_csv.exists()
    if _created_fixture:
        fixture_csv.write_text(csv_content)

    patch_target = f"{__name__}.get_llm"

    try:
        # ── Test 1: data_scientist_view returns only task + manifest_detail
        view = data_scientist_view(fake_state, fake_task, "some schema")
        assert set(view.keys()) == {"task", "manifest_detail"}
        print("PASS: data_scientist_view returns only task and manifest_detail")

        # ── Test 2: successful pandas query ───────────────────────
        fake_llm_output = json.dumps({
            "query_type":  "pandas",
            "query":       "df[df['clearance_level'] == 'A'][['full_name', 'clearance_level']]",
            "explanation": "Filter employees with clearance level A",
        })
        mock_response = MagicMock()
        mock_response.content = fake_llm_output

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(patch_target, return_value=mock_llm):
            result: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

        assert result["success"] is True,  f"Expected success, got error: {result['error']}"
        assert result["task_id"]     == "t1"
        assert result["worker_type"] == "data_scientist"
        assert result["error"]       is None

        output = json.loads(result["output"])
        assert output["row_count"]  == 2,   f"Expected 2 rows, got {output['row_count']}"
        assert output["table_name"] == "employees.csv"
        names = [r["full_name"] for r in output["result_value"]]
        assert "Noa Levi"  in names
        assert "Yael Ben"  in names
        assert "Dan Cohen" not in names
        print(f"PASS: pandas query returned {output['row_count']} rows: {names}")

        # ── Test 3: missing table file returns success=False ───────
        # Use a filename guaranteed not to exist regardless of test data state
        bad_task: Task = {**fake_task, "source_id": "ghost_table"}
        mock_raw_entry = {
            "id": "ghost_table", "filename": "ghost_table_does_not_exist.csv",
            "type": "csv", "table_name": "ghost_table_does_not_exist",
        }
        with patch(patch_target, return_value=mock_llm), \
             patch(f"{__name__}._get_raw_entry", return_value=mock_raw_entry), \
             patch(f"{__name__}.get_manifest_detail", return_value="fake schema"):
            result_missing: TaskResult = asyncio.run(
                data_scientist_worker(fake_state, bad_task)
            )

        assert result_missing["success"] is False
        assert "not found" in result_missing["error"]
        print("PASS: missing table file returns success=False with clear error")

        # ── Test 4: bad LLM JSON raises ValueError ─────────────────
        bad_response = MagicMock()
        bad_response.content = "I cannot answer that."
        mock_llm.ainvoke = AsyncMock(return_value=bad_response)

        with patch(patch_target, return_value=mock_llm):
            try:
                asyncio.run(data_scientist_worker(fake_state, fake_task))
                assert False, "Should have raised ValueError"
            except ValueError as exc:
                assert "I cannot answer that." in str(exc)
                print("PASS: ValueError raised with raw output on bad LLM JSON")

        # ── Test 5: bad query execution is sandboxed ───────────────
        crash_output = json.dumps({
            "query_type":  "pandas",
            "query":       "df['nonexistent_column_xyz']",
            "explanation": "This query will fail",
        })
        mock_response.content = crash_output
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(patch_target, return_value=mock_llm):
            result_bad: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

        assert result_bad["success"] is False
        assert result_bad["error"] is not None
        print("PASS: bad query execution is sandboxed — returns success=False, no crash")

    finally:
        # Only remove the file if this function created it (standalone run).
        # Under pytest the conftest session fixture owns the lifecycle.
        if _created_fixture and fixture_csv.exists():
            fixture_csv.unlink()

    print("\nPASS: all data_scientist tests passed")


if __name__ == "__main__":
    test_data_scientist()
