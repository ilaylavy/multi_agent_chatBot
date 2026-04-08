"""
core/registry.py — Worker Registry.

WORKER_REGISTRY maps worker_type strings to async worker callables.
Adding a new data type = add one entry here + write the agent.
Planner uses only string names. Router handles dispatch via get_worker().
Neither ever needs to change when a new worker is added.
"""

from __future__ import annotations

import inspect
import logging
from typing import Callable

from agents.librarian import librarian_worker
from agents.data_scientist import data_scientist_worker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WORKER_REGISTRY: dict[str, Callable] = {
    "librarian":      librarian_worker,
    "data_scientist": data_scientist_worker,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_worker(worker_type: str) -> Callable:
    """
    Return the async worker callable for the given worker_type.

    Parameters
    ----------
    worker_type : str
        Must be a key in WORKER_REGISTRY (e.g. 'librarian', 'data_scientist').

    Returns
    -------
    Callable
        Async function with signature (state: AgentState, task: Task) -> TaskResult.

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
# Isolated test — run with: python -m core.registry
# ---------------------------------------------------------------------------

def test_registry():
    # 1. Both required keys must exist
    assert "librarian"      in WORKER_REGISTRY, "Missing key: librarian"
    assert "data_scientist" in WORKER_REGISTRY, "Missing key: data_scientist"
    print("PASS: both keys present in WORKER_REGISTRY")

    # 2. Registry points to the real worker functions (not placeholders)
    assert WORKER_REGISTRY["librarian"]      is librarian_worker
    assert WORKER_REGISTRY["data_scientist"] is data_scientist_worker
    print("PASS: registry points to real librarian_worker and data_scientist_worker")

    # 3. Workers are async callables with correct (state, task, ...) signature
    for name in ("librarian", "data_scientist"):
        worker = get_worker(name)
        assert callable(worker),                    f"{name} is not callable"
        assert inspect.iscoroutinefunction(worker), f"{name} must be async"
        params = list(inspect.signature(worker).parameters.keys())
        assert params[:2] == ["state", "task"],     f"{name} signature wrong: {params}"
    print("PASS: both workers are async with (state, task, ...) signature")

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
