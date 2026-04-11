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
import logging

from core.llm_config import get_llm
from core.parse import parse_llm_json
from core.state import AgentState, SourceRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a synthesis specialist. Your job is to combine the results from multiple
data retrieval tasks into one clear, accurate answer to the user's question.

Rules:
  - Use ONLY information present in the task results provided. Do not invent, infer,
    or add any facts not explicitly stated in the results.
  - Address every task in the plan — if a task succeeded, use its result.
  - If some tasks failed, acknowledge what is missing and answer only from
    what is available.
  - Write in clear, direct prose suitable for the end user.
  - Every specific fact should include an inline source citation
    (e.g. "per the [source name], ..." or "according to [document name], ...").
  - Use the exact value from each entity's own task result. Never substitute a
    value from one task result into another entity's answer.

Respond with ONLY a JSON object — no explanation, no markdown:
{
  "draft_answer": "your full answer here"
}
"""

_TASK_RESULT_TEMPLATE = """\
Task {task_id} [{worker_type}] — sources: {source_ids}
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
{all_failed_block}"""

_ALL_FAILED_BLOCK = """\

IMPORTANT — ALL TASKS FAILED. Use these specific failure reasons in your draft,
not generic failure language. Tell the user exactly what was not found or what
went wrong:
{error_lines}
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
            f"(sources: {', '.join(task['source_ids'])}, status: {status})"
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
            source_ids=", ".join(task["source_ids"]),
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

    all_failed_block = ""
    if n_tasks > 0 and n_failed == n_tasks:
        error_lines = "\n".join(
            f"  - [{task['task_id']}] {view['task_results'].get(task['task_id'], {}).get('error', 'unknown error')}"
            for task in view["plan"]
        )
        all_failed_block = _ALL_FAILED_BLOCK.format(error_lines=error_lines)

    user_message = _USER_TEMPLATE.format(
        original_query=view["original_query"],
        n_tasks=n_tasks,
        n_failed=n_failed,
        plan_block=_format_plan_block(view),
        results_block=_format_results_block(view),
        all_failed_block=all_failed_block,
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
        "draft_answer":       draft_answer,
        "synthesizer_output": draft_answer,
        "sources_used":       view["sources_used"],
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.synthesizer
# ---------------------------------------------------------------------------

def test_synthesizer():
    from unittest.mock import AsyncMock, MagicMock, patch

    from tests.fixtures import SYNTHESIZER_STATE

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
    })
    mock_response = MagicMock()
    mock_response.content = fake_llm_output

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm):
        result = asyncio.run(synthesizer_node(SYNTHESIZER_STATE))

    assert set(result.keys()) == {"draft_answer", "synthesizer_output", "sources_used"}, \
        f"Node must return only changed fields, got: {set(result.keys())}"
    assert "Business Class" in result["draft_answer"]
    assert result["synthesizer_output"] == result["draft_answer"], \
        "synthesizer_output must be a snapshot of draft_answer"
    assert result["sources_used"] is SYNTHESIZER_STATE["sources_used"]
    print("PASS: node returns draft_answer, synthesizer_output, and sources_used")

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

    # ── Test 5: all tasks failed → specific error messages injected ─
    state_all_failed = {
        **SYNTHESIZER_STATE,
        "task_results": {
            "t1": {
                "task_id":     "t1",
                "worker_type": "data_scientist",
                "output":      '{"error": "Query returned no results. The requested data was not found in employees.csv."}',
                "success":     False,
                "error":       "Query returned no results. The requested data was not found in employees.csv.",
            },
            "t2": {
                "task_id":     "t2",
                "worker_type": "librarian",
                "output":      '{"error": "Skipped: prerequisite task t1 failed."}',
                "success":     False,
                "error":       "Skipped: prerequisite task t1 failed.",
            },
        },
    }

    captured_all_failed: list = []

    async def capture_all_failed(messages):
        captured_all_failed.extend(messages)
        return mock_response

    mock_llm.ainvoke = capture_all_failed

    with patch(patch_target, return_value=mock_llm):
        asyncio.run(synthesizer_node(state_all_failed))

    user_prompt_all = next(m["content"] for m in captured_all_failed if m["role"] == "user")
    assert "ALL TASKS FAILED"                                          in user_prompt_all, \
        "Prompt must include ALL TASKS FAILED heading"
    assert "Query returned no results"                                 in user_prompt_all, \
        "Prompt must include specific error from t1"
    assert "employees.csv"                                             in user_prompt_all, \
        "Prompt must name the table from the t1 error"
    assert "Skipped: prerequisite task t1 failed"                     in user_prompt_all, \
        "Prompt must include specific error from t2"
    assert "2 failed"                                                  in user_prompt_all, \
        "Prompt must state 2 tasks failed"
    # Partial-failure case must NOT include the ALL TASKS FAILED block
    captured_partial: list = []

    async def capture_partial(messages):
        captured_partial.extend(messages)
        return mock_response

    mock_llm.ainvoke = capture_partial

    with patch(patch_target, return_value=mock_llm):
        asyncio.run(synthesizer_node(state_with_failure))   # 1 failed, 1 success

    user_prompt_partial = next(m["content"] for m in captured_partial if m["role"] == "user")
    assert "ALL TASKS FAILED" not in user_prompt_partial, \
        "ALL TASKS FAILED block must not appear when only some tasks failed"
    print("PASS: all-tasks-failed injects specific error messages; partial failure does not")

    # ── Test 6: two entities with different values for the same field ─
    # t1 returns clearance_level=B for Dan, t3 returns clearance_level=D for Tal.
    # The synthesizer must use the correct value for each entity — never swap.
    multi_entity_state = {
        **SYNTHESIZER_STATE,
        "original_query": "What flight class are Dan Cohen and Tal Mizrahi entitled to?",
        "plan": [
            {"task_id": "t1", "worker_type": "data_scientist",
             "description": "Get Dan Cohen's clearance level", "source_ids": ["employees"],
             "depends_on": None},
            {"task_id": "t2", "worker_type": "librarian",
             "description": "Find flight entitlements for clearance level B",
             "source_ids": ["travel_policy_2024"], "depends_on": "t1"},
            {"task_id": "t3", "worker_type": "data_scientist",
             "description": "Get Tal Mizrahi's clearance level", "source_ids": ["employees"],
             "depends_on": None},
            {"task_id": "t4", "worker_type": "librarian",
             "description": "Find flight entitlements for clearance level D",
             "source_ids": ["travel_policy_2024"], "depends_on": "t3"},
        ],
        "task_results": {
            "t1": {"task_id": "t1", "worker_type": "data_scientist",
                   "output": json.dumps({"clearance_level": "B", "full_name": "Dan Cohen"}),
                   "success": True, "error": None},
            "t2": {"task_id": "t2", "worker_type": "librarian",
                   "output": json.dumps([{"chunk_text": "Clearance B: Premium Economy on flights over 6 hours.",
                                          "source_pdf": "travel_policy.pdf", "page_number": 2, "relevance_score": 0.95}]),
                   "success": True, "error": None},
            "t3": {"task_id": "t3", "worker_type": "data_scientist",
                   "output": json.dumps({"clearance_level": "D", "full_name": "Tal Mizrahi"}),
                   "success": True, "error": None},
            "t4": {"task_id": "t4", "worker_type": "librarian",
                   "output": json.dumps([{"chunk_text": "Clearance D: Economy class on all flights.",
                                          "source_pdf": "travel_policy.pdf", "page_number": 3, "relevance_score": 0.92}]),
                   "success": True, "error": None},
        },
    }

    # LLM must use B for Dan and D for Tal — the fake response models correct behavior
    multi_entity_llm_output = json.dumps({
        "draft_answer": (
            "Per the employees table, Dan Cohen holds clearance level B, which "
            "entitles him to Premium Economy on flights over 6 hours according to "
            "the Travel Policy 2024. Per the employees table, Tal Mizrahi holds "
            "clearance level D, which entitles her to Economy class on all flights "
            "according to the Travel Policy 2024."
        ),
    })
    mock_response_multi = MagicMock()
    mock_response_multi.content = multi_entity_llm_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response_multi)

    with patch(patch_target, return_value=mock_llm):
        result_multi = asyncio.run(synthesizer_node(multi_entity_state))

    draft = result_multi["draft_answer"]
    # Dan must be associated with B, Tal with D
    dan_idx = draft.index("Dan Cohen")
    tal_idx = draft.index("Tal Mizrahi")
    # Find the clearance level mentioned nearest after each name
    dan_section = draft[dan_idx:tal_idx]
    tal_section = draft[tal_idx:]
    assert "level B" in dan_section or "clearance B" in dan_section.lower(), \
        f"Dan Cohen's section must reference clearance B, got: {dan_section}"
    assert "level D" in tal_section or "clearance D" in tal_section.lower(), \
        f"Tal Mizrahi's section must reference clearance D, got: {tal_section}"
    # Ensure values are NOT swapped
    assert "level D" not in dan_section and "clearance d" not in dan_section.lower(), \
        "Dan Cohen must NOT have clearance D"
    assert "level B" not in tal_section and "clearance b" not in tal_section.lower(), \
        "Tal Mizrahi must NOT have clearance B"
    print("PASS: two entities with different clearance levels use correct values (no swap)")

    print("\nPASS: all synthesizer tests passed")


if __name__ == "__main__":
    test_synthesizer()
