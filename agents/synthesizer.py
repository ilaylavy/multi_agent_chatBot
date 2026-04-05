"""
agents/synthesizer.py — Synthesizer node (The Assembler).

LangGraph node: async synthesizer_node(state) -> dict

Combines all worker TaskResults into one coherent draft answer.
Rule: must not add any information not present in the task results.

View    : original_query, plan, task_results, sources_used
Returns : { draft_answer: str, sources_used: List[SourceRef] }
"""

from __future__ import annotations

import asyncio
import json

from core.llm_config import get_llm
from core.parse import parse_llm_json
from core.state import AgentState, SourceRef


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a synthesis specialist. Your job is to combine the results from multiple
data retrieval tasks into one clear, accurate answer to the user's question.

Rules (strictly enforced):
  - Use ONLY information present in the task results provided. Do not invent, infer,
    or add any facts not explicitly stated in the results.
  - Address every task in the plan — if a task succeeded, use its result.
  - If some tasks failed, acknowledge what is missing and answer only from
    what is available.
  - Write in clear, direct prose suitable for a business user.
  - Every specific fact MUST include an explicit inline source citation
    (e.g. "per the employees table, ..." or "according to the Travel Policy 2024, ...").
    An answer without inline citations will be rejected by the auditor.

Respond with ONLY a JSON object matching this schema — no explanation, no markdown:
{
  "draft_answer": "your full answer here",
  "confidence":   "high" | "medium" | "low"
}

confidence guide:
  high   — all tasks succeeded and results fully answer the question
  medium — some tasks failed or results are partial
  low    — most tasks failed or results are insufficient to answer
"""

_TASK_RESULT_TEMPLATE = """\
Task {task_id} [{worker_type}] — source: {source_id}
Description: {description}
Status: {status}
Result:
{result_body}
"""

_USER_TEMPLATE = """\
USER QUESTION:
{original_query}

PLAN ({n_tasks} tasks, {n_failed} failed):
{plan_block}

TASK RESULTS:
{results_block}
"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def synthesizer_view(state: AgentState) -> dict:
    """
    Returns only the fields the Synthesizer's LLM prompt may see.
    Does NOT include conversation_history, audit_result, retry_count, or retry_notes.
    """
    return {
        "original_query": state["original_query"],
        "plan":           state["plan"],
        "task_results":   state["task_results"],
        "sources_used":   state["sources_used"],
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_plan_block(view: dict) -> str:
    lines = []
    for task in view["plan"]:
        result = view["task_results"].get(task["task_id"], {})
        failed = result.get("success") is False
        status = "FAILED" if failed else "succeeded"
        lines.append(
            f"  - [{task['task_id']}] {task['description']} "
            f"(source: {task['source_id']}, status: {status})"
        )
    return "\n".join(lines)


def _format_results_block(view: dict) -> str:
    blocks = []
    for task in view["plan"]:
        task_id = task["task_id"]
        result  = view["task_results"].get(task_id, {})
        failed  = result.get("success") is False

        if failed:
            status      = "FAILED"
            result_body = f"ERROR: {result.get('error', 'unknown error')}"
        else:
            status = "SUCCESS"
            # Pretty-print the output JSON if possible, else use raw
            raw_output = result.get("output", json.dumps(result))
            try:
                parsed = json.loads(raw_output)
                result_body = json.dumps(parsed, indent=2)
            except (json.JSONDecodeError, TypeError):
                result_body = str(raw_output)

        blocks.append(_TASK_RESULT_TEMPLATE.format(
            task_id=task_id,
            worker_type=task["worker_type"],
            source_id=task["source_id"],
            description=task["description"],
            status=status,
            result_body=result_body,
        ))
    return "\n---\n".join(blocks)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def synthesizer_node(state: AgentState) -> dict:
    view = synthesizer_view(state)

    n_tasks  = len(view["plan"])
    n_failed = sum(
        1 for task in view["plan"]
        if view["task_results"].get(task["task_id"], {}).get("success") is False
    )

    user_message = _USER_TEMPLATE.format(
        original_query=view["original_query"],
        n_tasks=n_tasks,
        n_failed=n_failed,
        plan_block=_format_plan_block(view),
        results_block=_format_results_block(view),
    )

    llm = get_llm("synthesizer")
    response = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    data = parse_llm_json(response.content)
    try:
        draft_answer = data["draft_answer"]
    except KeyError as exc:
        raise ValueError(
            f"Missing key in LLM output: {exc}\nRaw output: {response.content}"
        ) from exc

    return {
        "draft_answer": draft_answer,
        "sources_used": view["sources_used"],
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.synthesizer
# ---------------------------------------------------------------------------

def test_synthesizer():
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fixtures import SYNTHESIZER_STATE

    patch_target = f"{__name__}.get_llm"

    # ── Test 1: synthesizer_view returns exactly the right fields ──
    view = synthesizer_view(SYNTHESIZER_STATE)
    assert set(view.keys()) == {"original_query", "plan", "task_results", "sources_used"}
    assert "conversation_history" not in view
    assert "audit_result"         not in view
    assert "retry_count"          not in view
    assert "retry_notes"          not in view
    print("PASS: synthesizer_view returns correct fields and excludes forbidden ones")

    # ── Test 2: node returns only draft_answer and sources_used ───
    fake_llm_output = json.dumps({
        "draft_answer": "Noa holds clearance level A, which entitles her to Business Class on flights over 4 hours per the Travel Policy 2024.",
        "confidence":   "high",
    })
    mock_response = MagicMock()
    mock_response.content = fake_llm_output

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm):
        result = asyncio.run(synthesizer_node(SYNTHESIZER_STATE))

    assert set(result.keys()) == {"draft_answer", "sources_used"}, \
        f"Node must return only changed fields, got: {set(result.keys())}"
    assert "Business Class" in result["draft_answer"]
    assert result["sources_used"] is SYNTHESIZER_STATE["sources_used"]
    print("PASS: node returns only draft_answer and sources_used")

    # ── Test 3: failed TaskResult appears explicitly in the prompt ─
    state_with_failure = {
        **SYNTHESIZER_STATE,
        "task_results": {
            "t1": {
                "task_id":     "t1",
                "worker_type": "data_scientist",
                "output":      '{"error": "Table file not found"}',
                "success":     False,
                "error":       "Table file not found: data/tables/employees.csv",
            },
            "t2": {
                "task_id":     "t2",
                "worker_type": "librarian",
                "output":      '[{"chunk_text": "Level A: Business Class", "source_pdf": "travel_policy.pdf", "page_number": 3, "relevance_score": 0.97}]',
                "success":     True,
                "error":       None,
            },
        },
    }

    captured_messages = []

    async def capture_invoke(messages):
        captured_messages.extend(messages)
        return mock_response

    mock_llm.ainvoke = capture_invoke

    with patch(patch_target, return_value=mock_llm):
        asyncio.run(synthesizer_node(state_with_failure))

    user_prompt = next(m["content"] for m in captured_messages if m["role"] == "user")
    assert "FAILED"                   in user_prompt, "Prompt must mention FAILED tasks"
    assert "1 failed"                 in user_prompt, "Prompt must state failure count"
    assert "Table file not found"     in user_prompt, "Prompt must include error message"
    assert "t1"                       in user_prompt
    print("PASS: failed TaskResult appears explicitly in the prompt with failure count and error")

    # ── Test 4: bad LLM JSON raises ValueError with raw output ────
    bad_response = MagicMock()
    bad_response.content = "Here is a summary of the results."
    mock_llm.ainvoke = AsyncMock(return_value=bad_response)

    with patch(patch_target, return_value=mock_llm):
        try:
            asyncio.run(synthesizer_node(SYNTHESIZER_STATE))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "Here is a summary of the results." in str(exc)
            print("PASS: ValueError raised with raw output on bad LLM JSON")

    print("\nPASS: all synthesizer tests passed")


if __name__ == "__main__":
    test_synthesizer()
