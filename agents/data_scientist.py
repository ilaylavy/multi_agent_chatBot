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
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.llm_config import _load_config, get_llm
from core.manifest import get_manifest_detail, get_manifest_detail_raw, get_manifest_details
from core.parse import parse_llm_json
from core.state import AgentState, Task, TaskResult

logger = logging.getLogger(__name__)

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
    raw = get_manifest_detail_raw()
    for section in ("pdfs", "tables"):
        for entry in raw.get(section, []):
            if entry["id"] == source_id:
                return entry
    raise ValueError(f"source_id '{source_id}' not found in manifest_detail.yaml")


def _make_json_safe(value: Any) -> Any:
    """
    Recursively convert numpy / pandas types to native Python types
    so that json.dumps never chokes on them.  NaN / NA values become None.
    """
    # NaN / NA checks first — before type-specific branches
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating,)) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value))            # "True" / "False"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_make_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    return value


def _execute_pandas(query: str, file_path: Path) -> tuple[Any, int]:
    """
    Load CSV into a DataFrame and eval the LLM-generated query expression.

    The query must reference `df` — e.g. `df[df['col'] == 'val']`.
    Sandboxed: __builtins__ is empty so import/exec/os are not accessible.

    Returns (result, row_count).  Scalar results are normalised to
    ``{"result_value": value, "row_count": 1}`` shape by the caller.
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
        # Scalar — bool, int, float, string, numpy scalar
        row_count = 1
        result_value = {"result_value": result, "row_count": 1}
    return _make_json_safe(result_value), row_count


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


def _resolve_file_path(raw_entry: dict) -> Path:
    """Resolve the on-disk path for a manifest entry."""
    filename = raw_entry.get("filename", "")
    base_path = raw_entry.get("base_path")
    if base_path:
        return _PROJECT_ROOT / base_path / filename
    return _tables_dir() / filename


def _load_into_memory_sqlite(raw_entries: list[dict]) -> tuple[sqlite3.Connection, list[str]]:
    """
    Load multiple sources into one in-memory SQLite database.

    CSV files are loaded via pandas read_csv → df.to_sql().
    SQLite files are ATTACHed and their tables copied in.
    Returns (connection, list_of_table_names).
    Raises ValueError on duplicate table names.
    """
    conn = sqlite3.connect(":memory:")
    table_names: list[str] = []

    for entry in raw_entries:
        source_type = entry.get("type", "").lower()
        file_path = _resolve_file_path(entry)
        table_name = entry.get("table_name", Path(entry.get("filename", "")).stem)

        if table_name in table_names:
            conn.close()
            raise ValueError(
                f"Duplicate table name '{table_name}' when loading multiple sources. "
                f"Cannot load both into the same in-memory database."
            )

        if source_type == "csv":
            df = pd.read_csv(file_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        elif source_type == "sqlite":
            conn.execute(f"ATTACH DATABASE '{file_path}' AS attached_db")
            cursor = conn.execute(
                "SELECT name FROM attached_db.sqlite_master WHERE type='table'"
            )
            for (tbl,) in cursor.fetchall():
                if tbl in table_names:
                    conn.execute("DETACH DATABASE attached_db")
                    conn.close()
                    raise ValueError(
                        f"Duplicate table name '{tbl}' when loading SQLite source "
                        f"'{entry['id']}'. Cannot load both into the same in-memory database."
                    )
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS [{tbl}] AS SELECT * FROM attached_db.[{tbl}]"
                )
                table_names.append(tbl)
            conn.execute("DETACH DATABASE attached_db")
            continue  # table_names already extended above for sqlite
        else:
            conn.close()
            raise ValueError(f"Unknown source type '{source_type}' for source '{entry['id']}'")

        table_names.append(table_name)

    return conn, table_names


def _execute_sql_conn(query: str, conn: sqlite3.Connection) -> tuple[Any, int]:
    """
    Execute a SQL SELECT query against an open SQLite connection.

    Only SELECT statements are permitted. Any other statement raises ValueError.
    Returns (rows_as_list_of_dicts, row_count).
    """
    stripped = query.strip().upper()
    if not stripped.startswith("SELECT"):
        raise ValueError(
            f"Only SELECT statements are permitted. Got: {query[:80]}"
        )
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query)
    rows = [dict(row) for row in cursor.fetchall()]
    return rows, len(rows)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a data query specialist. You will be given a task and the full schema
of one or more data tables. Your job is to reason about the data and write a
precise query that retrieves exactly what the task requires.

Reasoning step (required): before writing the query, think through which
tables are involved, how they relate to each other, what join keys connect
them (if multiple tables), and what the query will return. Write this
reasoning first in your response JSON.

Rules:
  - For a single CSV table: write a pandas expression using `df` as the
    DataFrame variable. The expression must evaluate to a result (scalar,
    list, or DataFrame). Do NOT write `df = pd.read_csv(...)` — df is
    already loaded.
  - For a single SQLite table: write a standard SQL SELECT statement.
  - When multiple tables are provided: all tables are loaded into a single
    SQLite database. Write SQL (not pandas). You may JOIN across tables.
    The available table names are listed in the schema block.
  - Select only the columns needed to answer the task — not all columns.
    For pandas: df[df['x'] == 'val'][['col_a', 'col_b']], not df[df['x'] == 'val'].
    For SQL: SELECT col_a, col_b FROM ..., not SELECT * FROM ...
    Example: if the task is "find the entity name", query only the name column.
  - When counting or grouping, always include the group label column in the
    result. The result must be interpretable without knowing which group each
    number belongs to.
    For pandas: df.groupby('[group_column]').size().reset_index(name='count'), not
    df.groupby('[group_column]').size() alone (which loses the label in some formats).
    For SQL: SELECT [group_column], COUNT(*) as count FROM ... GROUP BY [group_column],
    not just SELECT COUNT(*) FROM ...
  - Never guess or invent values — only query what is in the table.
  - Before writing any query, verify that every column name you plan to use exists
    in the schema provided. If a needed column does not exist in the schema, set
    query to an empty string and explain which column is missing and what columns
    are available.
  - If the task cannot be answered from the table(s), set query to an empty string
    and explain why in the explanation field.

Pandas examples (use these patterns — replace column_a, column_b with actual column names from the schema):
  Filter rows:       df[df['column_a'] == 'category_value'][['column_a', 'column_b']]
  Get max value:     df['column_b'].max()
  Get row with max:  df.loc[df['column_b'].idxmax()][['column_a', 'column_b']]
  Sort and first N:  df.sort_values('column_b', ascending=False).head(3)
  Count unique:      df['column_a'].nunique()
  Group aggregate:   df.groupby('column_a')['column_b'].mean()
  Count per group:   df.groupby('column_a').size().reset_index(name='count')

Respond with ONLY a JSON object matching this schema — no explanation, no markdown:
{
  "reasoning": {
    "tables_used": ["table_name"],
    "relationships": "how tables relate (or null if single table)",
    "join_keys": ["column used to join"] or [],
    "result_description": "what the query will return"
  },
  "query_type":  "pandas" or "sql",
  "query":       "the exact query string to execute",
  "explanation": "one sentence describing what this query retrieves"
}
"""

_USER_TEMPLATE = """\
TABLE SCHEMA(S):
{manifest_details}

TASK:
{task_description}
"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def data_scientist_view(state: AgentState, task: Task, manifest_details: str) -> dict:
    """
    Returns only the assigned task and manifest details for its table(s).
    The LLM prompt is built from this view — nothing else from state is visible.
    """
    return {
        "task":             task,
        "manifest_details": manifest_details,
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

    Supports single-source (preserves pandas-for-CSV / SQL-for-SQLite) and
    multi-source (loads everything into in-memory SQLite, forces SQL mode).
    """
    source_ids = task["source_ids"]

    # ── Extract injected prerequisite context (if any) ────────────
    _PREREQ_MARKER = "\n\n[Prerequisite result from"
    desc = task["description"]
    injected_context = desc[desc.index(_PREREQ_MARKER):] if _PREREQ_MARKER in desc else None

    # ── Manifest ──────────────────────────────────────────────────
    manifest_details = get_manifest_details(source_ids)
    raw_entries      = [_get_raw_entry(sid) for sid in source_ids]
    tables_loaded    = [e.get("table_name", e.get("filename", "?")) for e in raw_entries]

    view = data_scientist_view(state, task, manifest_details)

    # ── File existence check — never guess numbers ────────────────
    for raw_entry in raw_entries:
        file_path = _resolve_file_path(raw_entry)
        if not file_path.exists():
            return TaskResult(
                task_id=task["task_id"],
                worker_type="data_scientist",
                output=json.dumps({"error": f"Table file not found: {file_path}"}),
                success=False,
                error=f"Table file not found: {file_path}. "
                      f"Place the file in {_tables_dir()} and retry.",
            )

    # ── Determine execution mode ─────────────────────────────────
    multi_source = len(source_ids) > 1

    # For single source, preserve exact original behavior
    if not multi_source:
        raw_entry    = raw_entries[0]
        source_type  = raw_entry.get("type", "").lower()
        filename     = raw_entry.get("filename", "")
        table_name   = raw_entry.get("table_name", filename)
        file_path    = _resolve_file_path(raw_entry)
    logger.debug(
        "[%s] DataScientist — table_names=%s",
        task["task_id"],
        [e.get("table_name", e.get("filename", "?")) for e in raw_entries],
    )

    # ── LLM query generation ──────────────────────────────────────
    user_message = _USER_TEMPLATE.format(
        manifest_details=view["manifest_details"],
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

    # Log reasoning for trace visibility (not stored in state)
    reasoning = data.get("reasoning")
    logger.debug(
        "[%s] DataScientist LLM parse — reasoning=%s query_type=%s query=%s",
        task["task_id"],
        json.dumps(reasoning, indent=2) if reasoning else "(none)",
        query_type,
        query,
    )

    # ── Pre-execution validation ─────────────────────────────────
    if multi_source:
        # Multi-source always uses SQL
        if query_type != "sql":
            logger.warning("Multi-source task got query_type='%s', forcing 'sql'", query_type)
            query_type = "sql"

    if query_type == "pandas":
        try:
            compile(query, "<string>", "eval")
        except SyntaxError as exc:
            return TaskResult(
                task_id=task["task_id"],
                worker_type="data_scientist",
                output=json.dumps({"error": f"Generated query has invalid syntax: {exc}. Query was: {query}",
                                   "query_used": query}),
                success=False,
                error=f"Generated query has invalid syntax: {exc}. Query was: {query}",
            )
    elif query_type == "sql":
        stripped_query = query.strip()
        if not stripped_query:
            return TaskResult(
                task_id=task["task_id"],
                worker_type="data_scientist",
                output=json.dumps({"error": "LLM generated an empty SQL query.", "query_used": query}),
                success=False,
                error="LLM generated an empty SQL query.",
            )
        if not stripped_query.upper().startswith("SELECT"):
            return TaskResult(
                task_id=task["task_id"],
                worker_type="data_scientist",
                output=json.dumps({"error": f"Only SELECT statements are permitted. Got: {query[:80]}",
                                   "query_used": query}),
                success=False,
                error=f"Only SELECT statements are permitted. Got: {query[:80]}",
            )

    # ── Sandboxed execution ───────────────────────────────────────
    try:
        if multi_source:
            # Multi-source: load all tables into in-memory SQLite
            conn, table_names = _load_into_memory_sqlite(raw_entries)
            try:
                result_value, row_count = _execute_sql_conn(query, conn)
            finally:
                conn.close()
            table_name = ", ".join(table_names)
        elif query_type == "pandas":
            result_value, row_count = _execute_pandas(query, file_path)
        elif query_type == "sql":
            result_value, row_count = _execute_sql(query, file_path)
        else:
            raise ValueError(f"Unknown query_type '{query_type}' — must be 'pandas' or 'sql'")
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for sandboxing
        logger.debug("[%s] DataScientist exec error — %s", task["task_id"], exc)
        return TaskResult(
            task_id=task["task_id"],
            worker_type="data_scientist",
            output=json.dumps({"error": str(exc), "query_used": query}),
            success=False,
            error=f"Query execution failed: {exc}",
        )

    logger.debug("[%s] DataScientist exec — row_count=%d", task["task_id"], row_count)

    if row_count == 0:
        return TaskResult(
            task_id=task["task_id"],
            worker_type="data_scientist",
            output=json.dumps({"error": f"Query returned no results. The requested data was not found in {table_name}."}),
            success=False,
            error=f"Query returned no results. The requested data was not found in {table_name}.",
        )

    output = json.dumps(_make_json_safe({
        "result_value":    result_value,
        "query_used":      query,
        "table_name":      table_name,
        "row_count":       row_count,
        "reasoning":       reasoning,
        "tables_loaded":   tables_loaded,
        "injected_context": injected_context,
    }))

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
        "source_ids":  ["employees"],
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

    # ── Fixture CSV lives in tests/fixtures/tables/ — committed to git, never deleted ──
    fixture_tables_dir = _PROJECT_ROOT / "tests" / "fixtures" / "tables"
    patch_target       = f"{__name__}.get_llm"
    tables_dir_patch   = f"{__name__}._tables_dir"

    # ── Test 1: data_scientist_view returns only task + manifest_details
    view = data_scientist_view(fake_state, fake_task, "some schema")
    assert set(view.keys()) == {"task", "manifest_details"}
    print("PASS: data_scientist_view returns only task and manifest_details")

    # ── Test 2: successful pandas query ───────────────────────
    fake_llm_output = json.dumps({
        "reasoning": {
            "tables_used": ["employees"],
            "relationships": None,
            "join_keys": [],
            "result_description": "Returns full_name and clearance_level for level-A employees",
        },
        "query_type":  "pandas",
        "query":       "df[df['clearance_level'] == 'A'][['full_name', 'clearance_level']]",
        "explanation": "Filter employees with clearance level A",
    })
    mock_response = MagicMock()
    mock_response.content = fake_llm_output

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result["success"] is True,  f"Expected success, got error: {result['error']}"
    assert result["task_id"]     == "t1"
    assert result["worker_type"] == "data_scientist"
    assert result["error"]       is None

    output = json.loads(result["output"])
    assert output["row_count"]  >= 2,   f"Expected at least 2 rows, got {output['row_count']}"
    assert output["table_name"] == "employees.csv"
    names = [r["full_name"] for r in output["result_value"]]
    assert "Noa Levi"  in names
    assert "Dan Cohen" not in names    # Dan is clearance B, must not appear
    print(f"PASS: pandas query returned {output['row_count']} rows: {names}")

    # ── Test 3: missing table file returns success=False ───────
    # Use a filename guaranteed not to exist regardless of test data state
    bad_task: Task = {**fake_task, "source_ids": ["ghost_table"]}
    mock_raw_entry = {
        "id": "ghost_table", "filename": "ghost_table_does_not_exist.csv",
        "type": "csv", "table_name": "ghost_table_does_not_exist",
    }
    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir), \
         patch(f"{__name__}._get_raw_entry", return_value=mock_raw_entry), \
         patch(f"{__name__}.get_manifest_details", return_value="fake schema"):
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

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        try:
            asyncio.run(data_scientist_worker(fake_state, fake_task))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "I cannot answer that." in str(exc)
            print("PASS: ValueError raised with raw output on bad LLM JSON")

    # ── Test 5: bad query execution is sandboxed ───────────────
    crash_output = json.dumps({
        "reasoning": {"tables_used": ["employees"], "relationships": None, "join_keys": [], "result_description": "test"},
        "query_type":  "pandas",
        "query":       "df['nonexistent_column_xyz']",
        "explanation": "This query will fail",
    })
    mock_response.content = crash_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result_bad: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result_bad["success"] is False
    assert result_bad["error"] is not None
    print("PASS: bad query execution is sandboxed — returns success=False, no crash")

    # ── Test 6: query that executes correctly but returns 0 rows ───
    empty_output = json.dumps({
        "reasoning": {"tables_used": ["employees"], "relationships": None, "join_keys": [], "result_description": "test"},
        "query_type":  "pandas",
        "query":       "df[df['clearance_level'] == 'Z']",  # no level-Z employees in fixture
        "explanation": "Filter employees with clearance level Z (none exist)",
    })
    mock_response.content = empty_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result_empty: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result_empty["success"] is False, \
        "Empty result must be treated as failure, not success"
    assert result_empty["error"] is not None
    assert "no results" in result_empty["error"].lower(), \
        f"Error message must mention 'no results', got: {result_empty['error']}"
    assert "employees.csv" in result_empty["error"], \
        f"Error message must name the table, got: {result_empty['error']}"
    print(f"PASS: empty query result returns success=False: {result_empty['error']}")

    # ── Test 7: scalar bool result serializes cleanly ───────────
    bool_output = json.dumps({
        "reasoning": {"tables_used": ["employees"], "relationships": None, "join_keys": [], "result_description": "test"},
        "query_type":  "pandas",
        "query":       "(df['clearance_level'] == 'A').any()",
        "explanation": "Check if any employee has clearance level A",
    })
    mock_response.content = bool_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result_bool: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result_bool["success"] is True, f"Expected success, got error: {result_bool['error']}"
    out_bool = json.loads(result_bool["output"])
    assert out_bool["row_count"] == 1, f"Scalar result must have row_count=1, got {out_bool['row_count']}"
    assert "result_value" in out_bool["result_value"], \
        f"Scalar result must be wrapped as {{result_value: ...}}, got {out_bool['result_value']}"
    assert out_bool["result_value"]["result_value"] == "True", \
        f"Bool scalar must serialize as string 'True', got {out_bool['result_value']['result_value']!r}"
    print(f"PASS: scalar bool serializes cleanly: {out_bool['result_value']}")

    # ── Test 8: numpy integer result serializes cleanly ───────
    int_output = json.dumps({
        "reasoning": {"tables_used": ["employees"], "relationships": None, "join_keys": [], "result_description": "test"},
        "query_type":  "pandas",
        "query":       "df['clearance_level'].nunique()",
        "explanation": "Count distinct clearance levels",
    })
    mock_response.content = int_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result_int: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result_int["success"] is True, f"Expected success, got error: {result_int['error']}"
    out_int = json.loads(result_int["output"])
    assert out_int["row_count"] == 1, f"Scalar result must have row_count=1, got {out_int['row_count']}"
    assert "result_value" in out_int["result_value"], \
        f"Scalar result must be wrapped as {{result_value: ...}}, got {out_int['result_value']}"
    rv = out_int["result_value"]["result_value"]
    assert isinstance(rv, int), f"Numpy integer must serialize as native int, got {type(rv).__name__}: {rv!r}"
    print(f"PASS: numpy integer serializes cleanly: {out_int['result_value']}")

    # ── Test 9: invalid pandas syntax returns success=False with syntax error ──
    syntax_output = json.dumps({
        "reasoning": {"tables_used": ["employees"], "relationships": None, "join_keys": [], "result_description": "test"},
        "query_type":  "pandas",
        "query":       "df[df['clearance_level'] == 'A'",  # missing closing bracket
        "explanation": "Broken filter",
    })
    mock_response.content = syntax_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result_syntax: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result_syntax["success"] is False, "Invalid syntax must return success=False"
    assert "invalid syntax" in result_syntax["error"].lower(), \
        f"Error must mention 'invalid syntax', got: {result_syntax['error']}"
    assert "df[df['clearance_level'] == 'A'" in result_syntax["error"], \
        "Error must include the original query"
    out_syntax = json.loads(result_syntax["output"])
    assert "query_used" in out_syntax, "Output must include query_used for debugging"
    print(f"PASS: invalid pandas syntax returns success=False: {result_syntax['error'][:80]}")

    # ── Test 10: empty SQL string returns success=False ───────────
    empty_sql_output = json.dumps({
        "reasoning": {"tables_used": ["employees"], "relationships": None, "join_keys": [], "result_description": "test"},
        "query_type":  "sql",
        "query":       "",
        "explanation": "Cannot answer from this table",
    })
    mock_response.content = empty_sql_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result_empty_sql: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result_empty_sql["success"] is False, "Empty SQL query must return success=False"
    assert "empty" in result_empty_sql["error"].lower(), \
        f"Error must mention 'empty', got: {result_empty_sql['error']}"
    print(f"PASS: empty SQL query returns success=False: {result_empty_sql['error']}")

    # ── Test 11: DataFrame with NaN serializes NaN as null ─────────
    # _make_json_safe must convert NaN/NA to None so json.dumps produces null
    assert _make_json_safe(float("nan")) is None, "float NaN must become None"
    assert _make_json_safe(np.nan)       is None, "np.nan must become None"
    assert _make_json_safe(pd.NA)        is None, "pd.NA must become None"

    # Full round-trip: a DataFrame result with a NaN cell
    nan_df_records = [
        {"full_name": "Noa Levi", "clearance_level": "A", "bonus": 500.0},
        {"full_name": "Dan Cohen", "clearance_level": "B", "bonus": float("nan")},
    ]
    safe_records = _make_json_safe(nan_df_records)
    serialized = json.dumps(safe_records)   # must not raise
    parsed = json.loads(serialized)
    assert parsed[0]["bonus"] == 500.0, f"Non-NaN value must survive, got {parsed[0]['bonus']}"
    assert parsed[1]["bonus"] is None, f"NaN must serialize as null, got {parsed[1]['bonus']!r}"
    print("PASS: NaN/NA values serialize as null in JSON output")

    # ── Test 12: group-by query returns department name + count ─────
    groupby_output = json.dumps({
        "reasoning": {"tables_used": ["employees"], "relationships": None, "join_keys": [], "result_description": "test"},
        "query_type":  "pandas",
        "query":       "df.groupby('department').size().reset_index(name='count')",
        "explanation": "Count employees per department",
    })
    mock_response.content = groupby_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm), \
         patch(tables_dir_patch, return_value=fixture_tables_dir):
        result_group: TaskResult = asyncio.run(data_scientist_worker(fake_state, fake_task))

    assert result_group["success"] is True, f"Expected success, got error: {result_group['error']}"
    out_group = json.loads(result_group["output"])
    rows = out_group["result_value"]
    assert isinstance(rows, list), f"Group-by result must be a list of records, got {type(rows).__name__}"
    assert len(rows) > 0, "Group-by must return at least one row"
    first = rows[0]
    assert "department" in first, \
        f"Group-by result must include the group label 'department', got keys: {list(first.keys())}"
    assert "count" in first, \
        f"Group-by result must include the count column, got keys: {list(first.keys())}"
    assert isinstance(first["count"], int), \
        f"Count must be an integer, got {type(first['count']).__name__}"
    depts = [r["department"] for r in rows]
    assert "Engineering" in depts, f"Expected Engineering in departments, got {depts}"
    print(f"PASS: group-by query returns {len(rows)} rows with department name + count: {rows}")

    # ── Test 13: multi-table JOIN via in-memory SQLite ───────────
    multi_task: Task = {
        "task_id":     "t_multi",
        "worker_type": "data_scientist",
        "description": "Find each employee's salary_max by joining employees and salary_bands on clearance_level and department",
        "source_ids":  ["employees", "salary_bands"],
    }
    # Mock raw entries pointing to fixture files
    mock_employees_entry = {
        "id": "employees", "filename": "employees.csv", "type": "csv",
        "table_name": "employees", "base_path": "tests/fixtures/tables",
    }
    mock_salary_entry = {
        "id": "salary_bands", "filename": "salary_bands.sqlite", "type": "sqlite",
        "table_name": "salary_bands", "base_path": "tests/fixtures/tables",
    }
    multi_raw_entries = {"employees": mock_employees_entry, "salary_bands": mock_salary_entry}

    join_llm_output = json.dumps({
        "reasoning": {
            "tables_used": ["employees", "salary_bands"],
            "relationships": "employees.clearance_level = salary_bands.clearance_level AND employees.department = salary_bands.department",
            "join_keys": ["clearance_level", "department"],
            "result_description": "Returns each employee's name with their salary_max",
        },
        "query_type":  "sql",
        "query":       "SELECT e.full_name, e.clearance_level, e.department, s.salary_max "
                       "FROM employees e JOIN salary_bands s "
                       "ON e.clearance_level = s.clearance_level AND e.department = s.department",
        "explanation": "Join employees with salary_bands to get salary_max per employee",
    })
    mock_response.content = join_llm_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    def mock_get_raw_entry_multi(source_id: str) -> dict:
        return multi_raw_entries[source_id]

    with patch(patch_target, return_value=mock_llm), \
         patch(f"{__name__}._get_raw_entry", side_effect=mock_get_raw_entry_multi), \
         patch(f"{__name__}.get_manifest_details", return_value="fake multi-table schema"):
        result_multi: TaskResult = asyncio.run(data_scientist_worker(fake_state, multi_task))

    assert result_multi["success"] is True, f"Multi-table JOIN failed: {result_multi['error']}"
    out_multi = json.loads(result_multi["output"])
    assert out_multi["row_count"] >= 1, f"Expected at least 1 row from JOIN, got {out_multi['row_count']}"
    first_row = out_multi["result_value"][0]
    assert "full_name" in first_row, f"Result must include full_name, got keys: {list(first_row.keys())}"
    assert "salary_max" in first_row, f"Result must include salary_max, got keys: {list(first_row.keys())}"
    assert "employees" in out_multi["table_name"] and "salary_bands" in out_multi["table_name"], \
        f"table_name must list both tables, got: {out_multi['table_name']}"
    print(f"PASS: multi-table JOIN returned {out_multi['row_count']} rows: {first_row}")

    print("\nPASS: all data_scientist tests passed")


if __name__ == "__main__":
    test_data_scientist()
