"""
agents/chat.py — Chat Agent node (The Face).

LangGraph node: async chat_node(state) -> dict

Handles three distinct execution paths:

  INITIAL ENTRY   — final_answer is empty and retry_count < 3.
                    Appends the user query to conversation_history.
                    No LLM call. Passes control to the Planner.

  FINAL DELIVERY  — final_answer is populated.
                    Calls LLM to format the answer with source references.
                    Appends user query + formatted answer to history.

  RETRY EXHAUSTION — final_answer is empty and retry_count >= 3.
                    Returns the failure message from config.yaml.
                    No LLM call.

View    : original_query, conversation_history, final_answer, final_sources
Returns : { conversation_history } on initial entry
          { conversation_history, final_answer } on delivery or exhaustion
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.llm_config import _load_config, get_llm
from core.state import AgentState, Message

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read failure message and max_attempts from config — no hardcoding
_retry_cfg       = _load_config()["retry"]
_FAILURE_MESSAGE: str = _retry_cfg["failure_message"]
_MAX_ATTEMPTS:    int = _retry_cfg["max_attempts"]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a helpful assistant delivering a verified answer to a user.
Your job is to present the answer clearly and professionally.

Rules:
  - Present the answer directly — do not hedge or add uncertainty.
  - Cite sources inline where relevant (e.g. "per the Travel Policy 2024, Section 1").
  - Keep the response concise: one to three short paragraphs maximum.
  - Do not reveal internal system details (plans, retries, task IDs, worker names).
  - Respond in plain prose — no JSON, no markdown headers, no bullet lists unless
    the answer genuinely benefits from a list.
"""

_USER_TEMPLATE = """\
USER QUESTION:
{original_query}

VERIFIED ANSWER (format this for the user):
{final_answer}

SOURCES:
{sources_block}
"""


# ---------------------------------------------------------------------------
# View function
# ---------------------------------------------------------------------------

def chat_view(state: AgentState) -> dict:
    """
    Returns only the fields the Chat Agent's LLM prompt may see.
    Never exposes plan, task_results, audit_result, retry_count, or retry_notes.
    """
    return {
        "original_query":       state["original_query"],
        "conversation_history": state["conversation_history"],
        "final_answer":         state["final_answer"],
        "final_sources":        state["final_sources"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_sources_block(final_sources: list) -> str:
    if not final_sources:
        return "  (none)"
    lines = []
    for src in final_sources:
        src_id   = src["source_id"]
        src_type = src["source_type"]
        label    = src["label"]
        lines.append(f"  - {src_id} [{src_type}] — {label}")
    return "\n".join(lines)


def _append_messages(
    history: list,
    query: str,
    answer: str | None = None,
) -> list[Message]:
    """Return a new history list with query (and optionally answer) appended."""
    updated = list(history)
    # Only append the user message if it's not already the last entry
    if not updated or updated[-1].get("content") != query:
        updated.append(Message(role="user", content=query))
    if answer is not None:
        updated.append(Message(role="assistant", content=answer))
    return updated


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def chat_node(state: AgentState) -> dict:
    view = chat_view(state)

    final_answer = view["final_answer"]
    retry_count  = state["retry_count"]  # control logic only — not passed to LLM prompt, intentional view exception

    # ── Path 1: RETRY EXHAUSTION ──────────────────────────────────
    if not final_answer and retry_count >= _MAX_ATTEMPTS:
        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            _FAILURE_MESSAGE,
        )
        return {
            "conversation_history": updated_history,
            "final_answer":         _FAILURE_MESSAGE,
        }

    # ── Path 2: INITIAL ENTRY ─────────────────────────────────────
    if not final_answer:
        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
        )
        return {"conversation_history": updated_history}

    # ── Path 3: FINAL DELIVERY ────────────────────────────────────
    user_message = _USER_TEMPLATE.format(
        original_query=view["original_query"],
        final_answer=final_answer,
        sources_block=_format_sources_block(view["final_sources"]),
    )

    llm = get_llm("chat")
    response = await llm.ainvoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    formatted_answer = response.content.strip()

    updated_history = _append_messages(
        view["conversation_history"],
        view["original_query"],
        formatted_answer,
    )

    return {
        "conversation_history": updated_history,
        "final_answer":         formatted_answer,
    }


# ---------------------------------------------------------------------------
# Isolated test — run with: python -m agents.chat
# ---------------------------------------------------------------------------

def test_chat():
    from pathlib import Path as _Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from tests.fixtures import CHAT_AGENT_STATE

    patch_target = f"{__name__}.get_llm"

    # ── Shared minimal state skeleton ────────────────────────────
    base_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-session-001",
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

    # ── Test 1: chat_view exposes only the four permitted fields ──
    view = chat_view(CHAT_AGENT_STATE)
    assert set(view.keys()) == {
        "original_query", "conversation_history", "final_answer", "final_sources"
    }
    assert "plan"          not in view
    assert "task_results"  not in view
    assert "audit_result"  not in view
    assert "retry_count"   not in view
    assert "retry_notes"   not in view
    print("PASS: chat_view returns correct fields and excludes all forbidden ones")

    # ── Test 2: INITIAL ENTRY — no LLM call, history updated ──────
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()

    with patch(patch_target, return_value=mock_llm):
        result_initial = asyncio.run(chat_node(base_state))

    mock_llm.ainvoke.assert_not_called()
    assert "conversation_history" in result_initial
    assert "final_answer" not in result_initial
    assert result_initial["conversation_history"][-1]["role"]    == "user"
    assert result_initial["conversation_history"][-1]["content"] == base_state["original_query"]
    print("PASS: initial entry appends user message, no LLM call, no final_answer returned")

    # ── Test 3: FINAL DELIVERY — LLM called, history has both turns
    formatted = "Yes, Noa holds clearance level A and is entitled to Business Class on flights over 4 hours, per Travel Policy 2024 Section 1."
    mock_response = MagicMock()
    mock_response.content = f"  {formatted}  "   # intentional whitespace to test .strip()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch(patch_target, return_value=mock_llm):
        result_delivery = asyncio.run(chat_node(CHAT_AGENT_STATE))

    mock_llm.ainvoke.assert_called_once()
    assert set(result_delivery.keys()) == {"conversation_history", "final_answer"}
    assert result_delivery["final_answer"] == formatted   # stripped
    history = result_delivery["conversation_history"]
    roles   = [m["role"] for m in history]
    assert "user"      in roles
    assert "assistant" in roles
    assert history[-1]["role"]    == "assistant"
    assert history[-1]["content"] == formatted
    print("PASS: final delivery calls LLM, strips response, appends both turns to history")

    # ── Test 4: RETRY EXHAUSTION — no LLM call, failure message ──
    exhausted_state = {**base_state, "retry_count": _MAX_ATTEMPTS}

    with patch(patch_target, return_value=mock_llm):
        mock_llm.ainvoke.reset_mock()
        result_exhausted = asyncio.run(chat_node(exhausted_state))

    mock_llm.ainvoke.assert_not_called()
    assert result_exhausted["final_answer"] == _FAILURE_MESSAGE
    assert result_exhausted["conversation_history"][-1]["content"] == _FAILURE_MESSAGE
    assert result_exhausted["conversation_history"][-1]["role"]    == "assistant"
    print(f"PASS: retry exhaustion returns failure message from config, no LLM call: '{_FAILURE_MESSAGE}'")

    # ── Test 5: initial entry does not duplicate existing user message
    state_with_history = {
        **base_state,
        "conversation_history": [
            {"role": "user", "content": "Can Noa fly Business Class?"}
        ],
    }
    with patch(patch_target, return_value=mock_llm):
        result_no_dup = asyncio.run(chat_node(state_with_history))

    user_messages = [
        m for m in result_no_dup["conversation_history"] if m["role"] == "user"
    ]
    assert len(user_messages) == 1, "User message must not be duplicated in history"
    print("PASS: initial entry does not duplicate an existing user message in history")

    print("\nPASS: all chat tests passed")


if __name__ == "__main__":
    test_chat()
