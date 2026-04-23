"""
agents/auditor.py — Auditor node (The Gatekeeper).

LangGraph node: async auditor_node(state) -> dict

Verifies the draft answer against the plan. Writes verdict to state.
The Auditor NEVER decides routing — that is the conditional edge in graph.py.

View    : original_query, plan, draft_answer, sources_used
Returns (PASS) : { audit_result, final_answer, final_sources }
Returns (FAIL) : { audit_result, retry_count, retry_notes }
"""

from __future__ import annotations

import asyncio
import json
import logging

from core.llm_config import get_llm
from core.parse import parse_llm_json
from core.prompt_capture import capture as capture_prompt, get_prompts_for_attempt
from core.state import AgentState, AuditResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a quality auditor. Verify the draft answer against the plan and task results.

Check all three of the following:

  1. COMPLETENESS — a task is addressed if its result contributed to the final
     answer, directly or through a dependent task.
  2. ACCURACY — every factual claim in the draft must match the task results.
     Compare specific values (numbers, categories, dates) in the draft against
     the corresponding task result output. If a value does not match, describe
     the mismatch STRUCTURALLY — identify the field or claim that is wrong,
     without quoting the actual or expected values themselves.
     A claim is also acceptable if it follows logically from a chain of task
     results, even without explicit source attribution in the text.
  3. NO UNSUPPORTED ASSERTIONS — the draft does not assert anything that cannot
     be derived from the task results. Speculation beyond the evidence fails.

Verdict rules:
  - PASS: all three checks pass. Notes must briefly confirm.
  - FAIL: one or more checks fail. Notes must describe exactly what is wrong.

CRITICAL — Notes sanitization:
  Notes must NEVER include specific data values from the task results
  (numbers, names, dates, amounts, categories, identifiers). Describe what
  is wrong STRUCTURALLY — which field is off, which task's result the draft
  misused, which entity the claim concerns — without quoting the value.
    - Bad:  "task result shows ... but draft states no limit"
    - Good: "the draft omits a reimbursement limit that is present in the
             task result for the relevant entity"
  Why: the Synthesizer reads these notes as retry feedback. If you embed
  concrete values, the Synthesizer treats your notes as a data source
  instead of the task results, which causes accuracy drift on retry.

Retry routing (required when verdict is FAIL):
  Choose retry_target based on WHERE the failure originates:
    "synthesizer" — the task results contain the information needed to
                    answer correctly, but the draft misused, omitted, or
                    misrepresented it. Same task results go back to the
                    Synthesizer — no re-retrieval.
    "planner"     — the task results themselves are incomplete, wrong, or
                    missing information that should have been retrieved.
                    The plan needs to change or workers need to re-run.
  When verdict is PASS, set retry_target to null.

Respond with ONLY JSON — no explanation, no markdown:
{
  "verdict":       "PASS" | "FAIL",
  "notes":         "brief confirmation if PASS, structural description if FAIL (no data values)",
  "failed_checks": [],
  "retry_target":  "planner" | "synthesizer" | null
}

failed_checks: zero or more of "COMPLETENESS", "ACCURACY", "NO_UNSUPPORTED_ASSERTIONS"
"""

_USER_TEMPLATE = """\
ORIGINAL QUESTION:
{original_query}

PLAN ({n_tasks} tasks):
{plan_block}

TASK RESULTS:
{results_block}

SOURCES USED:
{sources_block}

DRAFT ANSWER:
{draft_answer}
"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def auditor_view(state: AgentState) -> dict:
    """
    Returns only the fields the Auditor's LLM prompt may see.
    When rewritten_query is non-empty it takes precedence over original_query so
    the Auditor verifies against the context-enriched version of the question.
    Does NOT include conversation_history, retry_count, or retry_notes.
    """
    return {
        "original_query": state.get("rewritten_query") or state["original_query"],
        "plan":           state["plan"],
        "task_results":   state["task_results"],
        "draft_answer":   state["draft_answer"],
        "sources_used":   state["sources_used"],
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_plan_block(plan: list) -> str:
    return "\n".join(
        f"  [{task['task_id']}] {task['description']} (sources: {', '.join(task['source_ids'])})"
        for task in plan
    )


def _format_results_block(plan: list, task_results: dict) -> str:
    lines = []
    for task in plan:
        task_id = task["task_id"]
        result = task_results.get(task_id, {})
        status = "FAILED" if result.get("success") is False else "SUCCESS"
        output = result.get("output", "(no output)")
        lines.append(f"  [{task_id}] {status}: {output}")
    return "\n".join(lines) if lines else "  (none)"


def _format_sources_block(sources_used: list) -> str:
    if not sources_used:
        return "  (none)"
    lines = []
    for src in sources_used:
        src_id   = src["source_id"]
        src_type = src["source_type"]
        label    = src["label"]
        lines.append(f"  - {src_id} ({src_type}) — {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def auditor_node(state: AgentState) -> dict:
    view = auditor_view(state)

    user_message = _USER_TEMPLATE.format(
        original_query=view["original_query"],
        n_tasks=len(view["plan"]),
        plan_block=_format_plan_block(view["plan"]),
        results_block=_format_results_block(view["plan"], view["task_results"]),
        sources_block=_format_sources_block(view["sources_used"]),
        draft_answer=view["draft_answer"],
    )

    llm = get_llm("auditor")
    attempt_number = state["retry_count"] + 1
    capture_prompt(
        state.get("session_id", ""),
        attempt_number,
        "auditor", "main",
        _SYSTEM_PROMPT, user_message,
    )
    response = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    data = parse_llm_json(response.content)
    try:
        verdict = data["verdict"]
        notes   = data.get("notes", "")
    except KeyError as exc:
        raise ValueError(
            f"Missing key in LLM output: {exc}\nRaw output: {response.content}"
        ) from exc

    # retry_target: planner (default) | synthesizer on FAIL; None on PASS.
    raw_target = data.get("retry_target")
    if verdict == "PASS":
        retry_target = None
    elif raw_target in ("planner", "synthesizer"):
        retry_target = raw_target
    else:
        retry_target = "planner"

    logger.debug(
        "[%s] Auditor — verdict=%s notes=%s attempt=%d",
        state.get("session_id", "?"),
        verdict,
        notes[:200] if notes else "(none)",
        state["retry_count"] + 1,
    )

    # Append to retry_history on every audit (PASS or FAIL).
    # agent_prompts carries every prompt captured during this attempt
    # (planner, workers, synthesizer, auditor). Empty list when tracing
    # is minimal or disabled — callers can treat it uniformly.
    history = list(state.get("retry_history", []))
    attempt_index = len(history) + 1
    history.append({
        "attempt":        attempt_index,
        "draft_answer":   view["draft_answer"],
        "audit_verdict":  verdict,
        "audit_notes":    notes,
        "retry_target":   retry_target,
        "agent_prompts":  get_prompts_for_attempt(
            state.get("session_id", ""), attempt_index,
        ),
    })

    if verdict == "PASS":
        return {
            "audit_result":  AuditResult(verdict="PASS", notes=notes, retry_target=None),
            "final_answer":  view["draft_answer"],
            "final_sources": view["sources_used"],
            "retry_history": history,
        }

    # FAIL — increment retry_count and write notes for the next agent.
    # Routing (→ Planner, → Synthesizer, or → Chat on exhaustion) is handled
    # by graph.py via retry_target — the Auditor only writes state.
    return {
        "audit_result":  AuditResult(verdict="FAIL", notes=notes, retry_target=retry_target),
        "retry_count":   state["retry_count"] + 1,  # control logic only — not passed to LLM prompt, intentional view exception
        "retry_notes":   notes,
        "retry_history": history,
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.auditor
# ---------------------------------------------------------------------------

def test_auditor():
    from unittest.mock import AsyncMock, MagicMock, patch

    from tests.fixtures import AUDITOR_STATE_FAIL, AUDITOR_STATE_PASS

    patch_target = f"{__name__}.get_llm"

    # ── Test 1: auditor_view returns exactly the right fields ──────
    view = auditor_view(AUDITOR_STATE_PASS)
    assert set(view.keys()) == {"original_query", "plan", "task_results", "draft_answer", "sources_used"}
    assert "conversation_history" not in view
    assert "retry_count"          not in view
    assert "retry_notes"          not in view
    print("PASS: auditor_view returns correct fields and excludes forbidden ones")

    # ── Test 2: PASS verdict — returns final_answer, final_sources, non-empty notes
    pass_notes = "All 2 tasks addressed. Claims supported by employees and travel_policy_2024 sources."
    pass_llm_output = json.dumps({
        "verdict":       "PASS",
        "notes":         pass_notes,
        "failed_checks": [],
    })
    mock_response = MagicMock()
    mock_response.content = pass_llm_output
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm):
        result_pass = asyncio.run(auditor_node(AUDITOR_STATE_PASS))

    assert "final_answer"  in result_pass
    assert "final_sources" in result_pass
    assert "audit_result"  in result_pass
    assert result_pass["audit_result"]["verdict"] == "PASS"
    assert result_pass["audit_result"]["notes"]   == pass_notes, \
        "PASS notes must carry the LLM-provided confirmation, not be empty"
    assert result_pass["audit_result"]["retry_target"] is None, \
        "PASS verdict must set retry_target to None"
    assert result_pass["final_answer"]  == AUDITOR_STATE_PASS["draft_answer"]
    assert result_pass["final_sources"] is AUDITOR_STATE_PASS["sources_used"]
    assert "retry_count" not in result_pass, "PASS must not touch retry_count"
    assert "retry_notes" not in result_pass, "PASS must not touch retry_notes"
    assert "retry_history" in result_pass, "PASS must include retry_history"
    assert len(result_pass["retry_history"]) == 1
    rh = result_pass["retry_history"][0]
    assert rh["attempt"]       == 1
    assert rh["audit_verdict"] == "PASS"
    assert rh["draft_answer"]  == AUDITOR_STATE_PASS["draft_answer"]
    assert rh["audit_notes"]   == pass_notes
    print("PASS: PASS verdict returns final_answer, final_sources, audit_result with non-empty notes, retry_history")

    # ── Test 3: FAIL verdict — increments retry_count, writes retry_notes
    fail_notes = (
        "Draft answer lacks source citations. "
        "Re-run and ensure every claim references a specific source."
    )
    fail_llm_output = json.dumps({
        "verdict":       "FAIL",
        "notes":         fail_notes,
        "failed_checks": ["ACCURACY", "NO_UNSUPPORTED_ASSERTIONS"],
        "retry_target":  "synthesizer",
    })
    mock_response.content = fail_llm_output

    with patch(patch_target, return_value=mock_llm):
        result_fail = asyncio.run(auditor_node(AUDITOR_STATE_FAIL))

    assert "audit_result" in result_fail
    assert "retry_count"  in result_fail
    assert "retry_notes"  in result_fail
    assert result_fail["audit_result"]["verdict"] == "FAIL"
    assert result_fail["audit_result"]["notes"]   == fail_notes
    assert result_fail["audit_result"]["retry_target"] == "synthesizer", \
        "FAIL verdict must carry the LLM-provided retry_target"
    assert result_fail["retry_history"][0]["retry_target"] == "synthesizer", \
        "retry_history entry must carry retry_target"
    assert result_fail["retry_count"] == AUDITOR_STATE_FAIL["retry_count"] + 1
    assert result_fail["retry_notes"] == fail_notes
    assert "final_answer"  not in result_fail, "FAIL must not set final_answer"
    assert "final_sources" not in result_fail, "FAIL must not set final_sources"
    assert "retry_history" in result_fail, "FAIL must include retry_history"
    assert len(result_fail["retry_history"]) == 1
    rh_fail = result_fail["retry_history"][0]
    assert rh_fail["attempt"]       == 1
    assert rh_fail["audit_verdict"] == "FAIL"
    assert rh_fail["draft_answer"]  == AUDITOR_STATE_FAIL["draft_answer"]
    assert rh_fail["audit_notes"]   == fail_notes
    print(f"PASS: FAIL verdict increments retry_count to {result_fail['retry_count']}, writes retry_notes, retry_history")

    # ── Test 4: Auditor never raises regardless of verdict ─────────
    for content, label in [
        (pass_llm_output, "PASS"),
        (fail_llm_output, "FAIL"),
    ]:
        mock_response.content = content
        try:
            with patch(patch_target, return_value=mock_llm):
                asyncio.run(auditor_node(AUDITOR_STATE_PASS))
        except Exception as exc:
            assert False, f"Auditor raised on {label} verdict: {exc}"
    print("PASS: Auditor never raises an exception regardless of verdict")

    # ── Test 5: bad LLM JSON raises ValueError with raw output ─────
    bad_response = MagicMock()
    bad_response.content = "The answer looks good to me."
    mock_llm.ainvoke = AsyncMock(return_value=bad_response)

    with patch(patch_target, return_value=mock_llm):
        try:
            asyncio.run(auditor_node(AUDITOR_STATE_PASS))
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "The answer looks good to me." in str(exc)
            print("PASS: ValueError raised with raw output on bad LLM JSON")

    # ── Test 6: prerequisite task used but not restated → PASS ──────
    # t1 looks up clearance level (intermediate), t2 uses it to find flight rules.
    # The draft answer mentions the flight entitlement but does NOT restate
    # "clearance level A" explicitly. The auditor must still PASS because t1's
    # result was consumed by t2 to produce the final answer.
    prereq_state = {
        **AUDITOR_STATE_PASS,
        "plan": [
            {"task_id": "t1", "worker_type": "data_scientist",
             "description": "Get Noa's clearance level", "source_ids": ["employees"],
             "depends_on": None},
            {"task_id": "t2", "worker_type": "librarian",
             "description": "Find flight class entitlements for the clearance level from t1",
             "source_ids": ["travel_policy_2024"],
             "depends_on": "t1"},
        ],
        "draft_answer": "Noa is entitled to Business Class on flights over 4 hours.",
    }

    prereq_pass_notes = "All 2 tasks addressed. t1 result used by t2 to determine entitlement."
    prereq_llm_output = json.dumps({
        "verdict":       "PASS",
        "notes":         prereq_pass_notes,
        "failed_checks": [],
    })
    mock_response.content = prereq_llm_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm):
        result_prereq = asyncio.run(auditor_node(prereq_state))

    assert result_prereq["audit_result"]["verdict"] == "PASS", \
        "Prerequisite task used but not restated must still PASS"
    assert "final_answer" in result_prereq, \
        "PASS must set final_answer"
    assert result_prereq["final_answer"] == prereq_state["draft_answer"]
    print("PASS: prerequisite task used but not explicitly restated returns PASS verdict")

    # ── Test 7: intermediate lookup without explicit source name → PASS ──
    # t1 gets Dan's clearance level from employees, t2 uses it to look up
    # flight entitlements from travel_policy_2024. The draft says
    # "Dan is entitled to Premium Economy" without saying "per the employees table".
    # This must PASS because the claim chains through t1 → t2.
    chain_state = {
        **AUDITOR_STATE_PASS,
        "plan": [
            {"task_id": "t1", "worker_type": "data_scientist",
             "description": "Get Dan Cohen's clearance level from employees",
             "source_ids": ["employees"], "depends_on": None},
            {"task_id": "t2", "worker_type": "librarian",
             "description": "Find flight entitlements for the clearance level from t1",
             "source_ids": ["travel_policy_2024"], "depends_on": "t1"},
        ],
        "sources_used": [
            {"source_id": "employees", "source_type": "csv", "label": "Employee records"},
            {"source_id": "travel_policy_2024", "source_type": "pdf", "label": "Travel Policy 2024"},
        ],
        "draft_answer": (
            "Dan Cohen is entitled to Premium Economy on international flights "
            "exceeding 6 hours. For shorter flights, Economy class applies."
        ),
    }

    chain_pass_notes = (
        "All 2 tasks addressed. Dan's clearance level from t1 was used by t2 "
        "to determine flight entitlement. Claims consistent with sources."
    )
    chain_llm_output = json.dumps({
        "verdict":       "PASS",
        "notes":         chain_pass_notes,
        "failed_checks": [],
    })
    mock_response.content = chain_llm_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm):
        result_chain = asyncio.run(auditor_node(chain_state))

    assert result_chain["audit_result"]["verdict"] == "PASS", \
        "Chained task results without explicit source names must PASS"
    assert result_chain["final_answer"] == chain_state["draft_answer"]
    print("PASS: chained intermediate lookup without explicit source name returns PASS")

    # ── Test 8: value mismatch between task result and draft → FAIL ──
    # t1 returns category=X but the draft states category=Y.
    # This must FAIL with ACCURACY in failed_checks.
    mismatch_state = {
        **AUDITOR_STATE_PASS,
        "plan": [
            {"task_id": "t1", "worker_type": "data_scientist",
             "description": "Look up the category for Entity A",
             "source_ids": ["records"], "depends_on": None},
        ],
        "task_results": {
            "t1": {"task_id": "t1", "worker_type": "data_scientist",
                   "output": '{"result_value": "X"}', "success": True, "error": None},
        },
        "sources_used": [
            {"source_id": "records", "source_type": "csv", "label": "Records table"},
        ],
        "draft_answer": "Entity A belongs to category Y.",
    }

    mismatch_notes = "The draft asserts a category for Entity A that does not match the task result."
    mismatch_llm_output = json.dumps({
        "verdict":       "FAIL",
        "notes":         mismatch_notes,
        "failed_checks": ["ACCURACY"],
        # retry_target intentionally omitted — tests default fallback to "planner"
    })
    mock_response.content = mismatch_llm_output
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm):
        result_mismatch = asyncio.run(auditor_node(mismatch_state))

    assert result_mismatch["audit_result"]["verdict"] == "FAIL", \
        "Value mismatch must return FAIL verdict"
    assert result_mismatch["audit_result"]["notes"] == mismatch_notes, \
        "FAIL notes must describe the value mismatch structurally"
    assert result_mismatch["audit_result"]["retry_target"] == "planner", \
        "Missing retry_target must default to 'planner' on FAIL"
    assert "retry_count" in result_mismatch, "FAIL must increment retry_count"
    assert "final_answer" not in result_mismatch, "FAIL must not set final_answer"
    print("PASS: value mismatch with missing retry_target defaults to 'planner'")

    print("\nPASS: all auditor tests passed")


if __name__ == "__main__":
    test_auditor()
