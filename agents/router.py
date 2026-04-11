"""
agents/router.py — Router node (The Dispatcher).

This IS a LangGraph node: async router_node(state) -> dict.
Internally it dispatches to worker callables — not to other graph nodes.

Dispatch flow:
  state["plan"] → get_worker() per task → asyncio.gather (parallel) →
  {task_results, sources_used, retrieved_chunks}

Rules:
  - Dispatch is always parallel via asyncio.gather — never a sequential loop.
  - Worker exceptions are caught and stored as failed TaskResults — never re-raised.
  - sources_used is built from successful TaskResults only.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from core.manifest import get_manifest_detail_raw, get_manifest_index_raw
from core.registry import get_worker
from core.state import AgentState, SourceRef, Task, TaskResult, router_view

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_source_refs(task: Task) -> list[SourceRef]:
    """
    Build a list of SourceRefs for a task by looking up the manifest.
    source_type is derived from the manifest detail section (pdfs → pdf,
    tables → entry["type"]).  label comes from the manifest index name field.
    """
    raw_detail = get_manifest_detail_raw()
    raw_index  = get_manifest_index_raw()
    refs: list[SourceRef] = []

    for source_id in task["source_ids"]:
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

        refs.append(SourceRef(source_id=source_id, source_type=source_type, label=label))

    return refs


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
    LangGraph node — dispatches tasks in dependency waves.

    Wave 1: tasks where depends_on is None — run in parallel via asyncio.gather.
    Wave N: tasks whose depends_on task_id completed in the previous wave — also
            run in parallel; the prerequisite output is appended to each task's
            description before dispatch.

    Returns only the changed fields: task_results, sources_used, retrieved_chunks.
    retrieved_chunks is populated from every successful Librarian result for RAGAS logging.
    """
    view = router_view(state)
    plan: list[Task] = view["plan"]

    task_results: dict[str, TaskResult] = {}

    # Group tasks by their depends_on value.
    # Wave order: None first, then resolve remaining tasks whose dep is now complete.
    remaining: list[Task] = list(plan)

    while remaining:
        # Find tasks whose prerequisite is already completed (or has no prerequisite)
        ready: list[Task] = [
            t for t in remaining
            if t.get("depends_on") is None or t.get("depends_on") in task_results
        ]

        if not ready:
            # Circular dependency or unresolvable — dispatch everything left to avoid deadlock
            logger.warning("Router: unresolvable dependency detected; dispatching remaining tasks")
            ready = remaining

        # Separate skipped tasks (prerequisite failed) from tasks to dispatch
        skipped: list[Task] = []
        enriched_tasks: list[Task] = []
        for task in ready:
            dep_id = task.get("depends_on")
            if dep_id and dep_id in task_results:
                prereq = task_results[dep_id]
                if not prereq["success"]:
                    skipped.append(task)
                    continue
                enriched = dict(task)
                enriched["description"] = (
                    task["description"]
                    + f"\n\n[Prerequisite result from {dep_id}]: {prereq['output']}"
                )
                enriched_tasks.append(enriched)  # type: ignore[arg-type]
            else:
                enriched_tasks.append(task)

        # Record skipped tasks immediately — do not call their workers
        for task in skipped:
            dep_id = task["depends_on"]
            task_results[task["task_id"]] = TaskResult(
                task_id=task["task_id"],
                worker_type=task["worker_type"],
                output=json.dumps({"error": f"Skipped: prerequisite task {dep_id} failed."}),
                success=False,
                error=f"Skipped: prerequisite task {dep_id} failed.",
            )

        # Parallel dispatch of this wave (only non-skipped tasks)
        if enriched_tasks:
            wave_results: list[Any] = await asyncio.gather(
                *(_dispatch_task(state, task) for task in enriched_tasks)
            )
            for result in wave_results:
                task_results[result["task_id"]] = result

        dispatched_ids = {t["task_id"] for t in ready}
        remaining = [t for t in remaining if t["task_id"] not in dispatched_ids]

    # Reconstruct ordered results list to match plan order for sources_used / chunks
    results: list[TaskResult] = [task_results[t["task_id"]] for t in plan]

    # Build sources_used from successful tasks only, deduplicated by source_id
    seen_source_ids: set[str] = set()
    sources_used: list[SourceRef] = []
    for task, result in zip(plan, results):
        if result["success"]:
            for ref in _build_source_refs(task):
                if ref["source_id"] not in seen_source_ids:
                    seen_source_ids.add(ref["source_id"])
                    sources_used.append(ref)

    # Unpack chunks from successful Librarian results for RAGAS logging.
    # Librarian output is a JSON array of Chunk dicts.
    retrieved_chunks: list = []
    for result in results:
        if result["success"] and result["worker_type"] == "librarian":
            try:
                chunks = json.loads(result["output"])
                if isinstance(chunks, list):
                    retrieved_chunks.extend(chunks)
            except (json.JSONDecodeError, TypeError):
                pass  # malformed output — skip silently

    return {
        "task_results":     task_results,
        "sources_used":     sources_used,
        "retrieved_chunks": retrieved_chunks,
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
        "source_ids":  ["travel_policy_2024"],
        "depends_on":  None,
    }
    ds_task: Task = {
        "task_id":     "t2",
        "worker_type": "data_scientist",
        "description": "Get Noa's clearance level from employees table",
        "source_ids":  ["employees"],
        "depends_on":  None,
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

    assert "task_results"     in result
    assert "sources_used"     in result
    assert "retrieved_chunks" in result
    assert set(result.keys()) == {"task_results", "sources_used", "retrieved_chunks"}, \
        "Node must return only changed fields"
    assert "t1" in result["task_results"]
    assert "t2" in result["task_results"]
    assert result["task_results"]["t1"] is lib_result
    assert result["task_results"]["t2"] is ds_result
    print("PASS: both task results stored, keyed by task_id")

    # ── retrieved_chunks populated from successful librarian output ─
    assert len(result["retrieved_chunks"]) > 0, \
        "retrieved_chunks must be non-empty after a successful librarian task"
    assert result["retrieved_chunks"][0]["chunk_text"] == "Level A: Business Class", \
        f"Unexpected chunk content: {result['retrieved_chunks'][0]}"
    print(f"PASS: retrieved_chunks populated — {len(result['retrieved_chunks'])} chunk(s) from librarian")

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

    # ── Test 6: dependency wave — t2 depends on t1 ────────────────
    # t1 runs first (alone); t2 runs second and receives t1's output in its description.
    dep_ds_task: Task = {
        "task_id":     "t1",
        "worker_type": "data_scientist",
        "description": "Get Noa's clearance level from employees table",
        "source_ids":  ["employees"],
        "depends_on":  None,
    }
    dep_lib_task: Task = {
        "task_id":     "t2",
        "worker_type": "librarian",
        "description": "Find flight class entitlements for the retrieved clearance level",
        "source_ids":  ["travel_policy_2024"],
        "depends_on":  "t1",
    }

    dep_state: AgentState = {**fake_state, "plan": [dep_ds_task, dep_lib_task]}

    dispatch_log: list[dict] = []   # records (task_id, description) at dispatch time

    async def logging_ds(state, task):
        dispatch_log.append({"task_id": task["task_id"], "description": task["description"]})
        return TaskResult(
            task_id="t1", worker_type="data_scientist",
            output='{"result_value": "A"}',
            success=True, error=None,
        )

    async def logging_lib(state, task):
        dispatch_log.append({"task_id": task["task_id"], "description": task["description"]})
        return TaskResult(
            task_id="t2", worker_type="librarian",
            output='[{"chunk_text": "Level A: Business Class"}]',
            success=True, error=None,
        )

    def dep_get_worker(worker_type: str):
        return logging_ds if worker_type == "data_scientist" else logging_lib

    with patch(patch_target, side_effect=dep_get_worker):
        result_dep = _asyncio.run(router_node(dep_state))

    # t1 must be dispatched before t2
    ids_in_order = [e["task_id"] for e in dispatch_log]
    assert ids_in_order == ["t1", "t2"], \
        f"Expected t1 dispatched before t2, got: {ids_in_order}"

    # t2's description must contain t1's output
    t2_desc = next(e["description"] for e in dispatch_log if e["task_id"] == "t2")
    assert '{"result_value": "A"}' in t2_desc, \
        f"t2 description must include t1 output, got: {t2_desc}"
    assert "[Prerequisite result from t1]" in t2_desc, \
        f"t2 description missing prerequisite header, got: {t2_desc}"

    # Both tasks complete successfully
    assert result_dep["task_results"]["t1"]["success"] is True
    assert result_dep["task_results"]["t2"]["success"] is True
    print(f"PASS: dependent task dispatched after prerequisite; output injected into description")

    # ── Test 7: prerequisite failed → dependent task skipped ──────
    skip_ds_task: Task = {
        "task_id":     "t1",
        "worker_type": "data_scientist",
        "description": "Get employee clearance level",
        "source_ids":  ["employees"],
        "depends_on":  None,
    }
    skip_lib_task: Task = {
        "task_id":     "t2",
        "worker_type": "librarian",
        "description": "Find policy for the retrieved clearance level",
        "source_ids":  ["travel_policy_2024"],
        "depends_on":  "t1",
    }

    skip_state: AgentState = {**fake_state, "plan": [skip_ds_task, skip_lib_task]}

    failed_prereq = TaskResult(
        task_id="t1", worker_type="data_scientist",
        output='{"error": "Query returned no results. The requested data was not found in employees.csv."}',
        success=False,
        error="Query returned no results. The requested data was not found in employees.csv.",
    )
    mock_failing_ds = AsyncMock(return_value=failed_prereq)
    mock_lib_spy    = AsyncMock()  # must NOT be called

    def skip_get_worker(worker_type: str):
        return mock_failing_ds if worker_type == "data_scientist" else mock_lib_spy

    with patch(patch_target, side_effect=skip_get_worker):
        result_skip = _asyncio.run(router_node(skip_state))

    # t1 ran and failed
    assert result_skip["task_results"]["t1"]["success"] is False

    # t2 must appear in task_results as skipped — worker must NOT have been called
    assert "t2" in result_skip["task_results"], \
        "Skipped task must still appear in task_results"
    t2_skipped = result_skip["task_results"]["t2"]
    assert t2_skipped["success"] is False
    assert "Skipped" in t2_skipped["error"], \
        f"Error must say 'Skipped', got: {t2_skipped['error']}"
    assert "t1" in t2_skipped["error"], \
        f"Error must name the failed prerequisite, got: {t2_skipped['error']}"
    mock_lib_spy.assert_not_called(), "Librarian worker must not be called when prerequisite failed"
    print(f"PASS: dependent task skipped when prerequisite failed; worker not called")

    # ── Test 8: duplicate source_id deduplicated in sources_used ────
    dup_task_1: Task = {
        "task_id": "t1", "worker_type": "data_scientist",
        "description": "Get Noa's clearance level",
        "source_ids": ["employees"], "depends_on": None,
    }
    dup_task_2: Task = {
        "task_id": "t2", "worker_type": "data_scientist",
        "description": "Get Noa's department",
        "source_ids": ["employees"], "depends_on": None,
    }
    dup_task_3: Task = {
        "task_id": "t3", "worker_type": "librarian",
        "description": "Find flight rules",
        "source_ids": ["travel_policy_2024"], "depends_on": None,
    }

    dup_state: AgentState = {**fake_state, "plan": [dup_task_1, dup_task_2, dup_task_3]}

    dup_ds_result_1 = TaskResult(task_id="t1", worker_type="data_scientist",
                                  output='{"result_value": "A"}', success=True, error=None)
    dup_ds_result_2 = TaskResult(task_id="t2", worker_type="data_scientist",
                                  output='{"result_value": "Engineering"}', success=True, error=None)
    dup_lib_result  = TaskResult(task_id="t3", worker_type="librarian",
                                  output='[{"chunk_text": "Business Class"}]', success=True, error=None)

    dup_results_map = {"t1": dup_ds_result_1, "t2": dup_ds_result_2, "t3": dup_lib_result}

    async def dup_worker(state, task):
        return dup_results_map[task["task_id"]]

    def dup_get_worker(worker_type: str):
        return dup_worker

    with patch(patch_target, side_effect=dup_get_worker):
        result_dup = _asyncio.run(router_node(dup_state))

    # All 3 tasks succeeded
    assert len(result_dup["task_results"]) == 3
    # sources_used must have only 2 entries (employees appears once, not twice)
    src_ids = [s["source_id"] for s in result_dup["sources_used"]]
    assert len(src_ids) == 2, \
        f"sources_used must deduplicate by source_id, got {len(src_ids)}: {src_ids}"
    assert src_ids.count("employees") == 1, \
        f"employees must appear exactly once, got {src_ids.count('employees')}"
    assert "travel_policy_2024" in src_ids
    print(f"PASS: duplicate source_id deduplicated in sources_used: {src_ids}")

    print("\nPASS: all router tests passed")


if __name__ == "__main__":
    test_router()
