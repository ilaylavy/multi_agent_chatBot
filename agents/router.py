"""
agents/router.py — Router node (The Dispatcher).

This IS a LangGraph node: async router_node(state) -> dict.
Internally it dispatches to worker callables — not to other graph nodes.

Dispatch flow:
  state["plan"] → get_worker() per task → asyncio.gather (parallel) →
  {task_results, sources_used}

Rules:
  - Dispatch is always parallel via asyncio.gather — never a sequential loop.
  - Worker exceptions are caught and stored as failed TaskResults — never re-raised.
  - sources_used is built from successful TaskResults only.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from core.manifest import _load_detail_raw, _load_index_raw
from core.registry import get_worker
from core.state import AgentState, SourceRef, Task, TaskResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_source_ref(task: Task) -> SourceRef:
    """
    Build a SourceRef for a task by looking up the manifest.
    source_type is derived from the manifest detail section (pdfs → pdf,
    tables → entry["type"]).  label comes from the manifest index name field.
    """
    source_id = task["source_id"]
    raw_detail = _load_detail_raw()
    raw_index  = _load_index_raw()

    # Determine source_type from which section the source lives in
    source_type = "unknown"
    for entry in raw_detail.get("pdfs", []):
        if entry["id"] == source_id:
            source_type = "pdf"
            break
    else:
        for entry in raw_detail.get("tables", []):
            if entry["id"] == source_id:
                source_type = entry.get("type", "csv")
                break

    # Get human-readable label from manifest index
    label = source_id  # fallback
    for section in ("pdfs", "tables"):
        for entry in raw_index.get(section, []):
            if entry["id"] == source_id:
                label = entry.get("name", source_id)
                break

    return SourceRef(source_id=source_id, source_type=source_type, label=label)


async def _dispatch_task(state: AgentState, task: Task) -> TaskResult:
    """
    Call the worker for a single task, catching any unhandled exception
    and converting it to a failed TaskResult so it never crashes the graph.
    """
    try:
        worker = get_worker(task["worker_type"])
        return await worker(state, task)
    except Exception as exc:  # noqa: BLE001
        return TaskResult(
            task_id=task["task_id"],
            worker_type=task["worker_type"],
            output=json.dumps({"error": str(exc)}),
            success=False,
            error=f"Worker raised an exception: {exc}",
        )


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def router_node(state: AgentState) -> dict:
    """
    LangGraph node — dispatches all tasks in the plan to workers in parallel.

    Returns only the changed fields: task_results and sources_used.
    """
    plan: list[Task] = state["plan"]

    # Parallel dispatch — asyncio.gather, never a sequential loop
    results: list[Any] = await asyncio.gather(
        *(_dispatch_task(state, task) for task in plan)
    )

    task_results: dict[str, TaskResult] = {
        result["task_id"]: result for result in results
    }

    # Build sources_used from successful tasks only
    sources_used: list[SourceRef] = [
        _build_source_ref(task)
        for task, result in zip(plan, results)
        if result["success"]
    ]

    return {
        "task_results": task_results,
        "sources_used": sources_used,
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.router
# ---------------------------------------------------------------------------

def test_router():
    import asyncio as _asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    lib_task: Task = {
        "task_id":     "t1",
        "worker_type": "librarian",
        "description": "Find flight class entitlements for clearance level A",
        "source_id":   "travel_policy_2024",
    }
    ds_task: Task = {
        "task_id":     "t2",
        "worker_type": "data_scientist",
        "description": "Get Noa's clearance level from employees table",
        "source_id":   "employees",
    }

    fake_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-session-001",
        "conversation_history": [],
        "plan":                 [lib_task, ds_task],
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

    lib_result = TaskResult(
        task_id="t1", worker_type="librarian",
        output='[{"chunk_text": "Level A: Business Class"}]',
        success=True, error=None,
    )
    ds_result = TaskResult(
        task_id="t2", worker_type="data_scientist",
        output='{"result_value": "A", "query_used": "df[...]", "table_name": "employees.csv", "row_count": 1}',
        success=True, error=None,
    )

    mock_lib_worker = AsyncMock(return_value=lib_result)
    mock_ds_worker  = AsyncMock(return_value=ds_result)

    def mock_get_worker(worker_type: str):
        return mock_lib_worker if worker_type == "librarian" else mock_ds_worker

    patch_target = f"{__name__}.get_worker"

    # ── Test 1: both task_results present, keyed by task_id ───────
    with patch(patch_target, side_effect=mock_get_worker):
        result = _asyncio.run(router_node(fake_state))

    assert "task_results" in result
    assert "sources_used" in result
    assert set(result.keys()) == {"task_results", "sources_used"}, \
        "Node must return only changed fields"
    assert "t1" in result["task_results"]
    assert "t2" in result["task_results"]
    assert result["task_results"]["t1"] is lib_result
    assert result["task_results"]["t2"] is ds_result
    print("PASS: both task results stored, keyed by task_id")

    # ── Test 2: both workers were actually called ──────────────────
    mock_lib_worker.assert_called_once()
    mock_ds_worker.assert_called_once()
    # Confirm state and correct task were passed to each
    _, lib_call_kwargs = mock_lib_worker.call_args
    _, ds_call_kwargs  = mock_ds_worker.call_args
    lib_call_args = mock_lib_worker.call_args[0]
    ds_call_args  = mock_ds_worker.call_args[0]
    assert lib_call_args[1]["task_id"] == "t1"
    assert ds_call_args[1]["task_id"]  == "t2"
    print("PASS: both workers called with correct state and task")

    # ── Test 3: asyncio.gather was used (not sequential calls) ────
    call_order: list[str] = []

    async def tracking_lib(state, task):
        call_order.append("lib_start")
        await _asyncio.sleep(0)   # yield to event loop
        call_order.append("lib_end")
        return lib_result

    async def tracking_ds(state, task):
        call_order.append("ds_start")
        await _asyncio.sleep(0)
        call_order.append("ds_end")
        return ds_result

    def tracking_get_worker(worker_type: str):
        return tracking_lib if worker_type == "librarian" else tracking_ds

    with patch(patch_target, side_effect=tracking_get_worker):
        _asyncio.run(router_node(fake_state))

    # With gather, both coroutines start before either finishes:
    # interleaved order: lib_start, ds_start, lib_end, ds_end (or similar)
    # Sequential would produce: lib_start, lib_end, ds_start, ds_end
    assert call_order.index("ds_start") < call_order.index("lib_end"), \
        f"Workers ran sequentially, not in parallel. Order: {call_order}"
    print(f"PASS: asyncio.gather confirmed — interleaved execution order: {call_order}")

    # ── Test 4: failed TaskResult is stored, not re-raised ────────
    failed_result = TaskResult(
        task_id="t1", worker_type="librarian",
        output='{"error": "collection not found"}',
        success=False, error="collection not found",
    )
    mock_lib_worker_fail = AsyncMock(return_value=failed_result)

    def mock_get_worker_partial(worker_type: str):
        return mock_lib_worker_fail if worker_type == "librarian" else mock_ds_worker

    with patch(patch_target, side_effect=mock_get_worker_partial):
        result_partial = _asyncio.run(router_node(fake_state))

    assert result_partial["task_results"]["t1"]["success"] is False
    assert result_partial["task_results"]["t2"]["success"] is True
    # sources_used only has the successful task
    assert len(result_partial["sources_used"]) == 1
    assert result_partial["sources_used"][0]["source_id"] == "employees"
    print("PASS: failed TaskResult stored without raising; absent from sources_used")

    # ── Test 5: unhandled worker exception is sandboxed ───────────
    mock_lib_crash = AsyncMock(side_effect=RuntimeError("unexpected crash"))

    def mock_get_worker_crash(worker_type: str):
        return mock_lib_crash if worker_type == "librarian" else mock_ds_worker

    with patch(patch_target, side_effect=mock_get_worker_crash):
        result_crash = _asyncio.run(router_node(fake_state))

    assert result_crash["task_results"]["t1"]["success"] is False
    assert "unexpected crash" in result_crash["task_results"]["t1"]["error"]
    print("PASS: worker exception sandboxed into failed TaskResult — graph never crashes")

    print("\nPASS: all router tests passed")


if __name__ == "__main__":
    test_router()
