"""
agents/planner.py — Planner agent.

Decomposes the user query into an ordered list of Tasks, each assigned
to a worker type (librarian | data_scientist) and a specific source_id.

View    : original_query, manifest_context, retry_notes (only on retry)
Returns : { plan: List[Task] }
"""

from __future__ import annotations

import asyncio
import json

from core.llm_config import get_llm
from core.parse import parse_llm_json
from core.state import AgentState, Task


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a query planner for a multi-agent RAG system.
Your job is to decompose the user's question into an ordered list of tasks.
Each task must be assigned to exactly one worker and one data source.

Available worker types:
  - librarian      : searches PDF documents (unstructured text)
  - data_scientist : queries CSV or SQLite tables (structured data)

You will be given:
  - The user's question.
  - A manifest listing all available data sources with their IDs and summaries.

Rules:
  - Only assign tasks to sources listed in the manifest.
  - Use the minimum number of tasks required to answer the question.
  - task_id values must be unique strings: t1, t2, t3, ...
  - worker_type must be exactly "librarian" or "data_scientist".
  - source_id must exactly match an id from the manifest.

Respond with ONLY a JSON object matching this schema — no explanation, no markdown:
{
  "tasks": [
    {
      "task_id": "t1",
      "worker_type": "librarian" | "data_scientist",
      "description": "what this task should find or retrieve",
      "source_id": "exact_id_from_manifest"
    }
  ]
}
"""

_RETRY_NOTE_SECTION = """\

RETRY CONTEXT — a previous answer was rejected. Use these notes to plan better:
{retry_notes}
"""

_USER_TEMPLATE = """\
AVAILABLE DATA SOURCES:
{manifest_context}

USER QUESTION:
{original_query}
{retry_section}"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def planner_view(state: AgentState) -> dict:
    """
    Returns only the fields the Planner's LLM prompt may see.
    retry_notes is included only when non-empty AND retry_count > 0.
    """
    view = {
        "original_query":   state["original_query"],
        "manifest_context": state["manifest_context"],
    }
    if state.get("retry_count", 0) > 0 and state.get("retry_notes", ""):
        view["retry_notes"] = state["retry_notes"]
    return view


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def planner_node(state: AgentState) -> dict:
    view = planner_view(state)

    retry_section = ""
    if "retry_notes" in view:
        retry_section = _RETRY_NOTE_SECTION.format(retry_notes=view["retry_notes"])

    user_message = _USER_TEMPLATE.format(
        manifest_context=view["manifest_context"],
        original_query=view["original_query"],
        retry_section=retry_section,
    )

    llm = get_llm("planner")
    response = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    data = parse_llm_json(response.content)
    try:
        tasks: list[Task] = [
            Task(
                task_id=t["task_id"],
                worker_type=t["worker_type"],
                description=t["description"],
                source_id=t["source_id"],
            )
            for t in data["tasks"]
        ]
    except KeyError as exc:
        raise ValueError(
            f"Missing key in LLM output: {exc}\nRaw output: {response.content}"
        ) from exc

    return {"plan": tasks}


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.planner
# ---------------------------------------------------------------------------

def test_planner():
    import os
    from unittest.mock import AsyncMock, MagicMock, patch

    from tests.fixtures import PLANNER_STATE, PLANNER_RETRY_STATE

    # ── Fake LLM response ────────────────────────────────────────
    fake_llm_output = json.dumps({
        "tasks": [
            {
                "task_id":     "t1",
                "worker_type": "data_scientist",
                "description": "Look up Noa's clearance level in the employee table",
                "source_id":   "employees",
            },
            {
                "task_id":     "t2",
                "worker_type": "librarian",
                "description": "Find flight class entitlements for clearance level A",
                "source_id":   "travel_policy_2024",
            },
        ]
    })

    mock_response = MagicMock()
    mock_response.content = fake_llm_output

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    # ── Test 1: view excludes retry_notes when retry_count == 0 ──
    view = planner_view(PLANNER_STATE)
    assert "retry_notes" not in view, "retry_notes must be absent when retry_count == 0"
    assert "original_query" in view
    assert "manifest_context" in view
    print("PASS: planner_view excludes retry_notes on first pass")

    # ── Test 2: view includes retry_notes on retry ────────────────
    retry_view = planner_view(PLANNER_RETRY_STATE)
    assert "retry_notes" in retry_view, "retry_notes must be present when retry_count > 0"
    print("PASS: planner_view includes retry_notes on retry")

    # patch target must match the module's __name__ at runtime:
    # "agents.planner" when imported, "__main__" when run directly
    patch_target = f"{__name__}.get_llm"

    # ── Test 3: node returns correct plan ────────────────────────
    with patch(patch_target, return_value=mock_llm):
        result = asyncio.run(planner_node(PLANNER_STATE))

    assert "plan" in result, "Result must contain 'plan'"
    assert len(result["plan"]) == 2
    assert result["plan"][0]["task_id"] == "t1"
    assert result["plan"][0]["worker_type"] == "data_scientist"
    assert result["plan"][0]["source_id"] == "employees"
    assert result["plan"][1]["task_id"] == "t2"
    assert result["plan"][1]["worker_type"] == "librarian"
    assert result["plan"][1]["source_id"] == "travel_policy_2024"
    assert set(result.keys()) == {"plan"}, "Node must return only changed fields"
    print(f"PASS: planner_node returns correct plan: {result['plan']}")

    # ── Test 4: bad JSON raises ValueError with raw output ────────
    bad_response = MagicMock()
    bad_response.content = "sorry, I cannot help with that"
    mock_llm.ainvoke = AsyncMock(return_value=bad_response)

    with patch(patch_target, return_value=mock_llm):
        try:
            asyncio.run(planner_node(PLANNER_STATE))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "sorry, I cannot help with that" in str(exc)
            print(f"PASS: ValueError raised with raw output on bad JSON")

    print("\nPASS: all planner tests passed")


if __name__ == "__main__":
    test_planner()
