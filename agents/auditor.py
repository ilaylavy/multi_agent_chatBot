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

from core.llm_config import get_llm
from core.parse import parse_llm_json
from core.state import AgentState, AuditResult


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a quality auditor for an AI answer system. Your job is to verify that a
draft answer meets strict accuracy and completeness standards before it reaches
a user.

Check all three of the following:

  1. COMPLETENESS — every task in the plan is addressed in the draft answer.
  2. TRACEABILITY — every factual claim in the draft answer is consistent with and
     derivable from the sources listed in sources_used. You are verifying factual
     accuracy, not citation formatting. A claim passes this check if it is supported
     by the available source data, regardless of whether the source name appears
     inline in the text.
  3. NO UNSUPPORTED ASSERTIONS — the draft does not assert anything that cannot
     be derived from the provided sources. Speculation or inference beyond the
     evidence fails this check.

Verdict rules:
  - PASS : all three checks pass. notes should be empty.
  - FAIL : one or more checks fail. notes must explain exactly what is wrong
           and what the Planner should do differently on retry.

Respond with ONLY a JSON object matching this schema — no explanation, no markdown:
{
  "verdict":       "PASS" | "FAIL",
  "notes":         "empty string if PASS, specific actionable feedback if FAIL",
  "failed_checks": []
}

failed_checks must be a list containing zero or more of:
  "COMPLETENESS", "TRACEABILITY", "NO_UNSUPPORTED_ASSERTIONS"
"""

_USER_TEMPLATE = """\
ORIGINAL QUESTION:
{original_query}

PLAN ({n_tasks} tasks — all must be addressed):
{plan_block}

SOURCES USED (claims must trace to one of these):
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
    Does NOT include conversation_history, task_results, retry_count, or retry_notes.
    """
    return {
        "original_query": state["original_query"],
        "plan":           state["plan"],
        "draft_answer":   state["draft_answer"],
        "sources_used":   state["sources_used"],
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_plan_block(plan: list) -> str:
    return "\n".join(
        f"  [{task['task_id']}] {task['description']} (source: {task['source_id']})"
        for task in plan
    )


def _format_sources_block(sources_used: list) -> str:
    if not sources_used:
        return "  (none)"
    lines = []
    for src in sources_used:
        # Support both SourceRef schema (source_id/source_type/label)
        # and the fixture schema (id/type) gracefully
        src_id   = src.get("source_id") or src.get("id",   "unknown")
        src_type = src.get("source_type") or src.get("type", "unknown")
        label    = src.get("label", src_id)
        page     = f" p.{src['page']}" if "page" in src else ""
        lines.append(f"  - {src_id} ({src_type}){page} — {label}")
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
        sources_block=_format_sources_block(view["sources_used"]),
        draft_answer=view["draft_answer"],
    )

    llm = get_llm("auditor")
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

    if verdict == "PASS":
        return {
            "audit_result":  AuditResult(verdict="PASS", notes=""),
            "final_answer":  view["draft_answer"],
            "final_sources": view["sources_used"],
        }

    # FAIL — increment retry_count and write notes for the Planner
    # Routing (→ Planner or → Chat on exhaustion) is handled by graph.py,
    # not here. The Auditor only writes state.
    return {
        "audit_result": AuditResult(verdict="FAIL", notes=notes),
        "retry_count":  state["retry_count"] + 1,  # control logic only — not passed to LLM prompt, intentional view exception
        "retry_notes":  notes,
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.auditor
# ---------------------------------------------------------------------------

def test_auditor():
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fixtures import AUDITOR_STATE_FAIL, AUDITOR_STATE_PASS

    patch_target = f"{__name__}.get_llm"

    # ── Test 1: auditor_view returns exactly the right fields ──────
    view = auditor_view(AUDITOR_STATE_PASS)
    assert set(view.keys()) == {"original_query", "plan", "draft_answer", "sources_used"}
    assert "conversation_history" not in view
    assert "task_results"         not in view
    assert "retry_count"          not in view
    assert "retry_notes"          not in view
    print("PASS: auditor_view returns correct fields and excludes forbidden ones")

    # ── Test 2: PASS verdict — returns final_answer and final_sources
    pass_llm_output = json.dumps({
        "verdict":       "PASS",
        "notes":         "",
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
    assert result_pass["audit_result"]["notes"]   == ""
    assert result_pass["final_answer"]  == AUDITOR_STATE_PASS["draft_answer"]
    assert result_pass["final_sources"] is AUDITOR_STATE_PASS["sources_used"]
    assert "retry_count" not in result_pass, "PASS must not touch retry_count"
    assert "retry_notes" not in result_pass, "PASS must not touch retry_notes"
    print("PASS: PASS verdict returns final_answer, final_sources, audit_result")

    # ── Test 3: FAIL verdict — increments retry_count, writes retry_notes
    fail_notes = (
        "Draft answer lacks source citations. "
        "Re-run and ensure every claim references a specific source."
    )
    fail_llm_output = json.dumps({
        "verdict":       "FAIL",
        "notes":         fail_notes,
        "failed_checks": ["TRACEABILITY", "NO_UNSUPPORTED_ASSERTIONS"],
    })
    mock_response.content = fail_llm_output

    with patch(patch_target, return_value=mock_llm):
        result_fail = asyncio.run(auditor_node(AUDITOR_STATE_FAIL))

    assert "audit_result" in result_fail
    assert "retry_count"  in result_fail
    assert "retry_notes"  in result_fail
    assert result_fail["audit_result"]["verdict"] == "FAIL"
    assert result_fail["audit_result"]["notes"]   == fail_notes
    assert result_fail["retry_count"] == AUDITOR_STATE_FAIL["retry_count"] + 1
    assert result_fail["retry_notes"] == fail_notes
    assert "final_answer"  not in result_fail, "FAIL must not set final_answer"
    assert "final_sources" not in result_fail, "FAIL must not set final_sources"
    print(f"PASS: FAIL verdict increments retry_count to {result_fail['retry_count']}, writes retry_notes")

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

    print("\nPASS: all auditor tests passed")


if __name__ == "__main__":
    test_auditor()
