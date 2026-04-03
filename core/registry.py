"""
core/registry.py — Worker Registry.

WORKER_REGISTRY maps worker_type strings to async node functions.
Adding a new data type = add one entry here + write the agent.
Planner uses only string names. Router handles dispatch via get_worker().
Neither ever needs to change when a new worker is added.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from core.state import AgentState, Task, TaskResult


# ---------------------------------------------------------------------------
# Placeholder worker functions
# Replaced in-place when agents/librarian.py and agents/data_scientist.py
# are built. Signature must stay: (state, task) -> TaskResult.
# ---------------------------------------------------------------------------

async def _librarian_placeholder(state: AgentState, task: Task) -> TaskResult:
    return TaskResult(
        task_id=task["task_id"],
        worker_type="librarian",
        output="[librarian placeholder — not yet implemented]",
        success=True,
        error=None,
    )


async def _data_scientist_placeholder(state: AgentState, task: Task) -> TaskResult:
    return TaskResult(
        task_id=task["task_id"],
        worker_type="data_scientist",
        output="[data_scientist placeholder — not yet implemented]",
        success=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WORKER_REGISTRY: dict[str, Callable] = {
    "librarian":      _librarian_placeholder,
    "data_scientist": _data_scientist_placeholder,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_worker(worker_type: str) -> Callable:
    """
    Return the async node function for the given worker_type.

    Parameters
    ----------
    worker_type : str
        Must be a key in WORKER_REGISTRY (e.g. 'librarian', 'data_scientist').

    Returns
    -------
    Callable
        The async function with signature (state: AgentState, task: Task) -> TaskResult.

    Raises
    ------
    ValueError
        If worker_type is not registered.
    """
    if worker_type not in WORKER_REGISTRY:
        known = list(WORKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown worker_type '{worker_type}'. "
            f"Registered workers: {known}"
        )
    return WORKER_REGISTRY[worker_type]


# ---------------------------------------------------------------------------
# Isolated test — run with: python core/registry.py
# ---------------------------------------------------------------------------

def test_registry():
    # 1. Both required keys must exist
    assert "librarian" in WORKER_REGISTRY, "Missing key: librarian"
    assert "data_scientist" in WORKER_REGISTRY, "Missing key: data_scientist"
    print("PASS: both keys present in WORKER_REGISTRY")

    # 2. get_worker with valid keys returns a callable
    for name in ("librarian", "data_scientist"):
        worker = get_worker(name)
        assert callable(worker), f"{name} is not callable"
    print("PASS: get_worker returns callable for both valid keys")

    # 3. Placeholder workers return correct TaskResult shape
    fake_state: AgentState = {
        "original_query":       "test query",
        "session_id":           "test-session",
        "conversation_history": [],
        "plan":                 [],
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

    fake_task: Task = {
        "task_id":     "t1",
        "worker_type": "librarian",
        "description": "Find travel policy sections",
        "source_id":   "travel_policy_2024",
    }

    for worker_type in ("librarian", "data_scientist"):
        fake_task["worker_type"] = worker_type
        worker = get_worker(worker_type)
        result: TaskResult = asyncio.run(worker(fake_state, fake_task))
        assert result["task_id"] == "t1"
        assert result["worker_type"] == worker_type
        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["output"], str)
    print("PASS: placeholder workers return valid TaskResult")

    # 4. Unknown worker_type raises ValueError with helpful message
    try:
        get_worker("nonexistent_worker")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "nonexistent_worker" in str(exc)
        assert "Registered workers:" in str(exc)
        print(f"PASS: ValueError raised correctly: {exc}")

    print("\nPASS: all registry tests passed")


if __name__ == "__main__":
    test_registry()
