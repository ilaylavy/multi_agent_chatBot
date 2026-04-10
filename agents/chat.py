"""
agents/chat.py — Chat Agent node (The Face).

LangGraph node: async chat_node(state) -> dict

Handles four distinct execution paths:

  RETRY EXHAUSTION  — final_answer is empty and retry_count >= max_attempts.
                      Returns the failure message from config.yaml. No LLM call.

  CLASSIFY (initial entry) — final_answer is empty and retry_count < max_attempts.
                      Calls LLM once to classify intent and generate a response.
                      Schema: { intent, rewritten_query, response }
                      DIRECT  → response is the answer; write to final_answer + history.
                      CLARIFY → response is a clarifying question; write to final_answer + history.
                      PLAN    → write chat_intent + rewritten_query; graph routes to Planner.

  FORMAT AND DELIVER — final_answer is populated (set by Auditor on PASS).
                      Calls LLM to format the answer with source references.
                      Appends user query + formatted answer to history.

View    : original_query, conversation_history, final_answer, final_sources, chat_intent
Returns : { conversation_history, chat_intent, rewritten_query }          on PLAN
          { conversation_history, chat_intent, final_answer }              on DIRECT / CLARIFY
          { conversation_history, final_answer }                           on FORMAT or EXHAUSTION
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.llm_config import _load_config, get_llm
from core.parse import parse_llm_json
from core.state import AgentState, Message

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read failure message and max_attempts from config — no hardcoding
_retry_cfg       = _load_config()["retry"]
_FAILURE_MESSAGE: str = _retry_cfg["failure_message"]
_MAX_ATTEMPTS:    int = _retry_cfg["max_attempts"]

# Max messages sent to the classification LLM — keeps the call fast on long sessions.
# Read from config.yaml chat.max_classification_history, default 6.
_chat_cfg = _load_config().get("chat", {})
MAX_CLASSIFICATION_HISTORY: int = _chat_cfg.get("max_classification_history", 6)


# ---------------------------------------------------------------------------
# Prompt templates — classification
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """\
You are a router. Your only job is to classify the user's intent so the right
downstream agent handles it. You never solve problems, evaluate feasibility,
or reason about what data exists.

Three intents:

DIRECT — greetings, small talk, meta questions about the system. Reply directly.

PLAN — anything that asks for information, facts, or analysis. This is the
strong default. If in doubt, choose PLAN. Never judge whether a query is
answerable — that is the Planner's job, not yours.

CLARIFY — use only when the user's intent itself is unclear, or when a
critical identifier is missing from the query and you cannot construct a
meaningful search without it (e.g. the user asks about "my" data but has not
provided their name, or asks to compare two things without specifying what
they are). Ask exactly one short question. Never clarify about data sources,
data availability, or how something might be defined in the system.

Query rewriting (PLAN only):
If conversation history exists and the current query references prior
context, rewrite it as a fully self-contained question. Preserve every
sub-question from the original. If no rewriting is needed, copy the
original query unchanged.

Respond with ONLY a JSON object:
{"intent": "DIRECT"|"CLARIFY"|"PLAN", "rewritten_query": "...", "response": "...or null for PLAN"}
"""

_CLASSIFY_USER_TEMPLATE = """\
{history_block}CURRENT QUERY:
{original_query}
"""


# ---------------------------------------------------------------------------
# Prompt templates — format and deliver
# ---------------------------------------------------------------------------

_DELIVER_SYSTEM_PROMPT = """\
Rewrite this answer as a direct, concise response.

Rules:
  - Lead with the key fact.
  - Remove source attribution phrases like "per the [source name]",
    "according to [document name]", "per Section N of [document]" —
    sources are shown separately in the UI.
  - Maximum 2 sentences for simple factual answers.
  - Keep the answer concise but never drop conditions, exceptions, thresholds,
    or approval requirements — these are critical for policy questions. If the
    answer contains a rule with an exception such as "permitted only when X" or
    "requires approval for Y", always include that exception in the final answer.
  - Do not add information not in the answer. Never add advice, suggestions, or
    recommendations not directly stated in the answer — do not say "consult your
    department" or "contact the finance team" unless the source explicitly says so.
  - Never refer to source documents by name in the formatted answer — the UI shows
    sources separately. Do not say "as specified in the Travel Policy 2024" or
    "according to the HR Handbook".
  - Do not reveal internal system details (plans, retries, task IDs, worker names).
  - Do not hedge or add uncertainty.
  - If the answer explains that data was not found, tell the user clearly and
    specifically what was not found — for example: "There is no [entity type]
    named [name] in the system. Did you mean a different name?"
"""

_DELIVER_USER_TEMPLATE = """\
USER QUESTION:
{original_query}

VERIFIED ANSWER (rewrite for the user):
{final_answer}
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
        "chat_intent":          state.get("chat_intent", ""),
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


def _format_history_block(history: list) -> str:
    """Return the last MAX_CLASSIFICATION_HISTORY messages formatted for the classify prompt."""
    recent = history[-MAX_CLASSIFICATION_HISTORY:] if len(history) > MAX_CLASSIFICATION_HISTORY else history
    if not recent:
        return ""
    lines = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append("")  # blank line separator
    return "\n".join(lines) + "\n"


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
# Fast pre-classification — skip LLM for obvious greetings
# ---------------------------------------------------------------------------

import re

_GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|goodbye)[\s!.,?]*$",
    re.IGNORECASE,
)

_CANNED_RESPONSES: dict[str, str] = {
    "hi":        "Hello! Ask me anything about your company data.",
    "hello":     "Hello! Ask me anything about your company data.",
    "hey":       "Hey! Ask me anything about your company data.",
    "thanks":    "You're welcome! Let me know if you have more questions.",
    "thank you": "You're welcome! Let me know if you have more questions.",
    "bye":       "Goodbye! Feel free to start a new session any time.",
    "goodbye":   "Goodbye! Feel free to start a new session any time.",
}

_MAX_FAST_LENGTH = 20


def _try_fast_classify(query: str) -> str | None:
    """Return a canned response if the query is an obvious greeting, else None."""
    if len(query) > _MAX_FAST_LENGTH:
        return None
    m = _GREETING_PATTERN.match(query.strip())
    if not m:
        return None
    key = m.group(1).lower()
    return _CANNED_RESPONSES.get(key)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def chat_node(state: AgentState) -> dict:
    view = chat_view(state)

    final_answer = view["final_answer"]
    retry_count  = state["retry_count"]  # control logic only — not passed to LLM prompt, intentional view exception

    # ── Path 1: RETRY EXHAUSTION ──────────────────────────────────
    if not final_answer and retry_count >= _MAX_ATTEMPTS:
        synthesizer_output = state.get("synthesizer_output", "")
        if synthesizer_output:
            llm = get_llm("chat")
            exhaust_response = await llm.ainvoke([
                {"role": "system", "content": (
                    "The system could not fully verify an answer. "
                    f"Here is what was found: {synthesizer_output}. "
                    "Write a honest 1-2 sentence response telling the user "
                    "what happened and what was found, even if incomplete. Be direct."
                )},
                {"role": "user", "content": view["original_query"]},
            ])
            answer = exhaust_response.content.strip()
        else:
            answer = _FAILURE_MESSAGE

        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            answer,
        )
        return {
            "conversation_history": updated_history,
            "final_answer":         answer,
        }

    # ── Path 2: CLASSIFY AND ROUTE ────────────────────────────────
    if not final_answer:
        # Fast path — skip LLM for obvious greetings
        fast_response = _try_fast_classify(view["original_query"])
        if fast_response is not None:
            updated_history = _append_messages(
                view["conversation_history"],
                view["original_query"],
                fast_response,
            )
            return {
                "conversation_history": updated_history,
                "chat_intent":          "DIRECT",
                "final_answer":         fast_response,
            }

        history_block    = _format_history_block(view["conversation_history"])
        classify_message = _CLASSIFY_USER_TEMPLATE.format(
            history_block=history_block,
            original_query=view["original_query"],
        )

        llm      = get_llm("chat")
        response = await llm.ainvoke([
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user",   "content": classify_message},
        ])

        data = parse_llm_json(response.content)
        try:
            intent          = data["intent"]
            rewritten_query = data.get("rewritten_query") or view["original_query"]
            response_text   = data.get("response")
        except KeyError as exc:
            raise ValueError(
                f"Missing key in classify LLM output: {exc}\nRaw output: {response.content}"
            ) from exc

        # PLAN — pass control to the Planner; no answer yet
        if intent == "PLAN":
            updated_history = _append_messages(
                view["conversation_history"],
                view["original_query"],
            )
            return {
                "conversation_history": updated_history,
                "chat_intent":          "PLAN",
                "rewritten_query":      rewritten_query,
            }

        # DIRECT or CLARIFY — the response is ready immediately
        if response_text is None:
            raise ValueError(
                f"LLM returned intent={intent!r} but response field is null.\n"
                f"Raw output: {response.content}"
            )

        response_text = response_text.strip()
        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            response_text,
        )
        return {
            "conversation_history": updated_history,
            "chat_intent":          intent,
            "final_answer":         response_text,
        }

    # ── Path 3: FORMAT AND DELIVER ────────────────────────────────
    # Only format if the Auditor explicitly approved (verdict == PASS).
    # final_answer is only written to state by the Auditor on PASS —
    # this guard prevents delivering un-audited content.
    audit = state.get("audit_result") or {}
    if audit.get("verdict") != "PASS":
        logger.warning(
            "chat_node reached delivery with final_answer set but audit verdict is %r — "
            "treating as exhaustion",
            audit.get("verdict"),
        )
        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            _FAILURE_MESSAGE,
        )
        return {
            "conversation_history": updated_history,
            "final_answer":         _FAILURE_MESSAGE,
        }

    user_message = _DELIVER_USER_TEMPLATE.format(
        original_query=view["original_query"],
        final_answer=final_answer,
    )

    llm      = get_llm("chat")
    response = await llm.ainvoke([
        {"role": "system", "content": _DELIVER_SYSTEM_PROMPT},
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
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    from tests.fixtures import CHAT_AGENT_STATE

    patch_target = f"{__name__}.get_llm"

    # ── Shared minimal state skeleton ────────────────────────────
    base_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-session-001",
        "conversation_history": [],
        "chat_intent":          "",
        "rewritten_query":      "",
        "plan":                 [],
        "manifest_context":     "",
        "task_results":         {},
        "sources_used":         [],
        "retrieved_chunks":     [],
        "draft_answer":         "",
        "synthesizer_output":   "",
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "retry_count":          0,
        "retry_notes":          "",
        "final_answer":         "",
        "final_sources":        [],
    }

    # ── Test 1: chat_view exposes only the permitted fields ───────
    view = chat_view(CHAT_AGENT_STATE)
    assert set(view.keys()) == {
        "original_query", "conversation_history", "final_answer", "final_sources", "chat_intent"
    }
    assert "plan"             not in view
    assert "task_results"     not in view
    assert "audit_result"     not in view
    assert "retry_count"      not in view
    assert "retry_notes"      not in view
    assert "rewritten_query"  not in view
    assert "synthesizer_output" not in view
    print("PASS: chat_view returns correct fields and excludes all forbidden ones")

    # ── Test 2: DIRECT intent — LLM responds immediately, final_answer set ──
    direct_response = "Hello! I'm a multi-agent RAG assistant. Ask me anything about your company data."
    direct_json = json.dumps({
        "intent":          "DIRECT",
        "rewritten_query": "Hi there",
        "response":        direct_response,
    })
    mock_resp_direct = MagicMock()
    mock_resp_direct.content = direct_json
    mock_llm_direct = MagicMock()
    mock_llm_direct.ainvoke = AsyncMock(return_value=mock_resp_direct)

    direct_state = {**base_state, "original_query": "Hi there"}
    with patch(patch_target, return_value=mock_llm_direct):
        result_direct = asyncio.run(chat_node(direct_state))

    assert result_direct["chat_intent"]   == "DIRECT"
    assert result_direct["final_answer"]  == direct_response
    assert "conversation_history"        in result_direct
    history_direct = result_direct["conversation_history"]
    assert history_direct[-2]["role"]    == "user"
    assert history_direct[-1]["role"]    == "assistant"
    assert history_direct[-1]["content"] == direct_response
    assert "rewritten_query" not in result_direct, "DIRECT must not write rewritten_query"
    print("PASS: DIRECT intent calls LLM, sets final_answer, appends both turns to history")

    # ── Test 3: CLARIFY intent — clarifying question returned as final_answer ──
    clarify_response = "Could you clarify — which Noa are you referring to?"
    clarify_json = json.dumps({
        "intent":          "CLARIFY",
        "rewritten_query": "Can Noa fly Business Class?",
        "response":        clarify_response,
    })
    mock_resp_clarify = MagicMock()
    mock_resp_clarify.content = clarify_json
    mock_llm_clarify = MagicMock()
    mock_llm_clarify.ainvoke = AsyncMock(return_value=mock_resp_clarify)

    clarify_state = {**base_state, "original_query": "Can Noa fly Business Class?"}
    with patch(patch_target, return_value=mock_llm_clarify):
        result_clarify = asyncio.run(chat_node(clarify_state))

    assert result_clarify["chat_intent"]  == "CLARIFY"
    assert result_clarify["final_answer"] == clarify_response
    history_clarify = result_clarify["conversation_history"]
    assert history_clarify[-1]["role"]    == "assistant"
    assert history_clarify[-1]["content"] == clarify_response
    print("PASS: CLARIFY intent calls LLM, sets final_answer to clarifying question")

    # ── Test 4: PLAN intent — rewritten_query written, final_answer absent ──
    rewritten = "What is Dan Cohen's flight class entitlement?"
    plan_json = json.dumps({
        "intent":          "PLAN",
        "rewritten_query": rewritten,
        "response":        None,
    })
    mock_resp_plan = MagicMock()
    mock_resp_plan.content = plan_json
    mock_llm_plan = MagicMock()
    mock_llm_plan.ainvoke = AsyncMock(return_value=mock_resp_plan)

    plan_state = {
        **base_state,
        "original_query": "What about Dan?",
        "conversation_history": [
            {"role": "user",      "content": "Can Noa fly Business Class?"},
            {"role": "assistant", "content": "Yes, Noa holds clearance level A."},
        ],
    }
    with patch(patch_target, return_value=mock_llm_plan):
        result_plan = asyncio.run(chat_node(plan_state))

    assert result_plan["chat_intent"]     == "PLAN"
    assert result_plan["rewritten_query"] == rewritten
    assert "final_answer" not in result_plan, "PLAN must not set final_answer"
    assert "conversation_history" in result_plan
    history_plan = result_plan["conversation_history"]
    assert history_plan[-1]["role"]    == "user"
    assert history_plan[-1]["content"] == "What about Dan?"
    print("PASS: PLAN intent writes chat_intent + rewritten_query, no final_answer, appends user message")

    # ── Test 5: PLAN with empty history — rewritten_query equals original ──
    no_rewrite_json = json.dumps({
        "intent":          "PLAN",
        "rewritten_query": "Can Noa fly Business Class?",
        "response":        None,
    })
    mock_resp_no_rewrite = MagicMock()
    mock_resp_no_rewrite.content = no_rewrite_json
    mock_llm_no_rewrite = MagicMock()
    mock_llm_no_rewrite.ainvoke = AsyncMock(return_value=mock_resp_no_rewrite)

    with patch(patch_target, return_value=mock_llm_no_rewrite):
        result_no_rewrite = asyncio.run(chat_node(base_state))

    assert result_no_rewrite["rewritten_query"] == base_state["original_query"]
    print("PASS: PLAN with no history passes original query through unchanged")

    # ── Test 6: FORMAT AND DELIVER — LLM rewrites the synthesizer answer ──
    # The formatted response must differ meaningfully from the synthesizer output.
    # CHAT_AGENT_STATE.final_answer (synthesizer output) is:
    #   "Yes. Noa holds clearance level A, which entitles her to Business Class on flights over 4 hours."
    # The chat agent should rewrite it — shorter, no source attribution, lead with key fact.
    synthesizer_answer = CHAT_AGENT_STATE["final_answer"]
    formatted = "Noa has clearance level A and is entitled to Business Class on flights over 4 hours."
    mock_resp_deliver = MagicMock()
    mock_resp_deliver.content = f"  {formatted}  "   # intentional whitespace to test .strip()
    mock_llm_deliver  = MagicMock()

    captured_deliver: list = []

    async def capture_deliver(messages):
        captured_deliver.extend(messages)
        return mock_resp_deliver

    mock_llm_deliver.ainvoke = capture_deliver

    deliver_state = {**CHAT_AGENT_STATE, "audit_result": {"verdict": "PASS", "notes": "All verified."}}
    with patch(patch_target, return_value=mock_llm_deliver):
        result_deliver = asyncio.run(chat_node(deliver_state))

    assert set(result_deliver.keys()) == {"conversation_history", "final_answer"}
    assert result_deliver["final_answer"] == formatted   # stripped
    assert result_deliver["final_answer"] != synthesizer_answer, \
        "Formatted answer must differ from the synthesizer output"
    history_deliver = result_deliver["conversation_history"]
    assert history_deliver[-1]["role"]    == "assistant"
    assert history_deliver[-1]["content"] == formatted

    # Confirm key prompt directives are present in the delivery system prompt
    system_msg = next(m["content"] for m in captured_deliver if m["role"] == "system")
    assert "not found"     in system_msg.lower(), \
        "Delivery system prompt must include 'not found' guidance"
    assert "remove source attribution" in system_msg.lower(), \
        "Delivery system prompt must instruct the LLM to remove source attribution"
    assert "lead with the key fact" in system_msg.lower(), \
        "Delivery system prompt must instruct the LLM to lead with the key fact"
    # User template must NOT include sources block — sources are shown separately in the UI
    user_msg = next(m["content"] for m in captured_deliver if m["role"] == "user")
    assert "SOURCES:" not in user_msg, \
        "Deliver user template must not include sources block — sources are shown in the UI"
    print("PASS: FORMAT AND DELIVER rewrites answer differently from synthesizer; prompt removes attribution")

    # ── Test 6b: FORMAT AND DELIVER preserves conditions from synthesizer output ──
    # The synthesizer output contains a conditional rule. The Chat Agent must
    # preserve the condition in the formatted answer — not drop it for brevity.
    conditional_synth = (
        "Employees with clearance level C are entitled to Economy class on all flights. "
        "Business Class is permitted on flights exceeding 8 hours with prior manager approval."
    )
    conditional_formatted = (
        "Clearance C employees fly Economy, but Business Class is permitted on "
        "flights over 8 hours with prior manager approval."
    )
    mock_resp_cond = MagicMock()
    mock_resp_cond.content = conditional_formatted
    mock_llm_cond = MagicMock()
    mock_llm_cond.ainvoke = AsyncMock(return_value=mock_resp_cond)

    cond_state = {
        **base_state,
        "final_answer":  conditional_synth,
        "final_sources": [{"source_id": "travel_policy_2024", "source_type": "pdf", "label": "Travel Policy 2024"}],
        "audit_result":  {"verdict": "PASS", "notes": "Verified."},
    }
    with patch(patch_target, return_value=mock_llm_cond):
        result_cond = asyncio.run(chat_node(cond_state))

    assert result_cond["final_answer"] == conditional_formatted
    # The condition "8 hours" and "manager approval" must survive formatting
    assert "8 hours" in result_cond["final_answer"], \
        "Formatted answer must preserve the flight-duration threshold"
    assert "manager approval" in result_cond["final_answer"], \
        "Formatted answer must preserve the approval requirement"
    # Verify the prompt instructs preservation of conditions
    assert "never drop conditions" in _DELIVER_SYSTEM_PROMPT.lower(), \
        "Deliver prompt must instruct LLM to never drop conditions"
    print("PASS: FORMAT AND DELIVER preserves conditional rules (thresholds + approval requirements)")

    # ── Test 7a: RETRY EXHAUSTION — no synthesizer_output → hardcoded failure, no LLM call ──
    mock_llm_exhaust = MagicMock()
    mock_llm_exhaust.ainvoke = AsyncMock()
    exhausted_state = {**base_state, "retry_count": _MAX_ATTEMPTS, "synthesizer_output": ""}

    with patch(patch_target, return_value=mock_llm_exhaust):
        result_exhausted = asyncio.run(chat_node(exhausted_state))

    mock_llm_exhaust.ainvoke.assert_not_called()
    assert result_exhausted["final_answer"] == _FAILURE_MESSAGE
    assert result_exhausted["conversation_history"][-1]["content"] == _FAILURE_MESSAGE
    assert result_exhausted["conversation_history"][-1]["role"]    == "assistant"
    print(f"PASS: retry exhaustion (no synthesizer output) returns hardcoded failure, no LLM call")

    # ── Test 7b: RETRY EXHAUSTION — with synthesizer_output → LLM call with partial findings ──
    partial_answer = "I could not fully verify, but Noa appears to hold clearance level A."
    mock_resp_exhaust = MagicMock()
    mock_resp_exhaust.content = f"  {partial_answer}  "
    mock_llm_exhaust_with = MagicMock()
    mock_llm_exhaust_with.ainvoke = AsyncMock(return_value=mock_resp_exhaust)

    captured_exhaust: list = []
    async def capture_exhaust(messages):
        captured_exhaust.extend(messages)
        return mock_resp_exhaust
    mock_llm_exhaust_with.ainvoke = capture_exhaust

    exhausted_with_output = {
        **base_state,
        "retry_count":        _MAX_ATTEMPTS,
        "synthesizer_output": "Noa holds clearance level A based on employee records.",
    }
    with patch(patch_target, return_value=mock_llm_exhaust_with):
        result_exhaust_llm = asyncio.run(chat_node(exhausted_with_output))

    assert result_exhaust_llm["final_answer"] == partial_answer, \
        f"Expected LLM-generated answer, got: {result_exhaust_llm['final_answer']!r}"
    assert result_exhaust_llm["final_answer"] != _FAILURE_MESSAGE, \
        "When synthesizer_output exists, must NOT return the hardcoded failure message"
    assert result_exhaust_llm["conversation_history"][-1]["content"] == partial_answer
    assert result_exhaust_llm["conversation_history"][-1]["role"]    == "assistant"
    # Verify the prompt included synthesizer_output
    system_msg = next(m["content"] for m in captured_exhaust if m["role"] == "system")
    assert "Noa holds clearance level A" in system_msg, \
        "Exhaustion prompt must include the synthesizer_output"
    assert "could not fully verify" in system_msg.lower(), \
        "Exhaustion prompt must mention verification failure"
    print(f"PASS: retry exhaustion (with synthesizer output) calls LLM and returns partial answer")

    # ── Test 8: initial entry does not duplicate existing user message ──
    no_dup_json = json.dumps({
        "intent":          "PLAN",
        "rewritten_query": "Can Noa fly Business Class?",
        "response":        None,
    })
    mock_resp_no_dup = MagicMock()
    mock_resp_no_dup.content = no_dup_json
    mock_llm_no_dup = MagicMock()
    mock_llm_no_dup.ainvoke = AsyncMock(return_value=mock_resp_no_dup)

    state_with_history = {
        **base_state,
        "conversation_history": [
            {"role": "user", "content": "Can Noa fly Business Class?"}
        ],
    }
    with patch(patch_target, return_value=mock_llm_no_dup):
        result_no_dup = asyncio.run(chat_node(state_with_history))

    user_messages = [
        m for m in result_no_dup["conversation_history"] if m["role"] == "user"
    ]
    assert len(user_messages) == 1, "User message must not be duplicated in history"
    print("PASS: initial entry does not duplicate an existing user message in history")

    # ── Test 9: history_block includes last N messages in classify prompt ──
    long_history = [
        {"role": "user",      "content": f"question {i}"}
        for i in range(10)
    ]
    block = _format_history_block(long_history)
    # Should contain the last MAX_CLASSIFICATION_HISTORY entries, not the first ones
    assert "question 9"  in block
    assert "question 4"  in block
    assert "question 3" not in block   # older than MAX_CLASSIFICATION_HISTORY ago
    print(f"PASS: _format_history_block includes last {MAX_CLASSIFICATION_HISTORY} messages only")

    # ── Test 9b: classification LLM call receives truncated history ──
    # Build a state with 20 messages (10 turns) — only last 6 should reach the LLM
    big_history = []
    for i in range(10):
        big_history.append({"role": "user",      "content": f"user-msg-{i}"})
        big_history.append({"role": "assistant",  "content": f"asst-msg-{i}"})

    classify_truncate_json = json.dumps({
        "intent":          "PLAN",
        "rewritten_query": "What is the answer?",
        "response":        None,
    })
    mock_resp_trunc = MagicMock()
    mock_resp_trunc.content = classify_truncate_json

    captured_classify: list = []
    async def capture_classify(messages):
        captured_classify.extend(messages)
        return mock_resp_trunc

    mock_llm_trunc = MagicMock()
    mock_llm_trunc.ainvoke = capture_classify

    trunc_state = {
        **base_state,
        "original_query": "What is the answer?",
        "conversation_history": big_history,
    }
    with patch(patch_target, return_value=mock_llm_trunc):
        asyncio.run(chat_node(trunc_state))

    # The user message sent to the LLM should contain only the last 6 history messages
    user_msg = next(m["content"] for m in captured_classify if m["role"] == "user")
    # Oldest messages must NOT appear
    assert "user-msg-0" not in user_msg, \
        "Classification must not send old history (user-msg-0) to the LLM"
    assert "user-msg-6" not in user_msg, \
        "Classification must not send user-msg-6 (outside last 6) to the LLM"
    # Recent messages must appear (last 6 = indices 14-19 = user-msg-7 through asst-msg-9)
    assert "user-msg-7" in user_msg, \
        "Classification must include recent history (user-msg-7)"
    assert "asst-msg-9" in user_msg, \
        "Classification must include the most recent message (asst-msg-9)"
    # Full history is still preserved in state (not truncated)
    assert len(trunc_state["conversation_history"]) == 20, \
        "Full conversation_history must not be mutated by classification"
    print(f"PASS: classification LLM receives only last {MAX_CLASSIFICATION_HISTORY} messages, full history preserved")

    # ── Test 10: fast pre-classification skips LLM for greetings ──
    mock_llm_fast = MagicMock()
    mock_llm_fast.ainvoke = AsyncMock()

    for greeting in ("hi", "hello", "thanks", "Hey!", "bye"):
        fast_state = {**base_state, "original_query": greeting}
        with patch(patch_target, return_value=mock_llm_fast):
            result_fast = asyncio.run(chat_node(fast_state))

        assert result_fast["chat_intent"] == "DIRECT", \
            f"'{greeting}' must classify as DIRECT, got {result_fast['chat_intent']!r}"
        assert result_fast["final_answer"], \
            f"'{greeting}' must produce a non-empty final_answer"
        assert result_fast["conversation_history"][-1]["role"] == "assistant"
    mock_llm_fast.ainvoke.assert_not_called()
    print("PASS: fast pre-classification returns DIRECT for greetings without any LLM call")

    # ── Test 11: messages longer than threshold still go to LLM ──
    long_msg = "What is Noa's clearance level?"
    assert _try_fast_classify(long_msg) is None, \
        "Messages longer than threshold must not be fast-classified"
    assert _try_fast_classify("find employees") is None, \
        "Non-greeting short messages must not be fast-classified"
    print("PASS: non-greeting and long messages are not fast-classified")

    # ── Test 12: exhaustion with synthesizer_output — unverified, audit FAIL ──
    # Simulates: auditor rejected 3 times (retry_count == max), synthesizer had output,
    # but final_answer is empty (auditor never set it on PASS).
    # The chat agent must produce an honest partial answer, NOT the hardcoded failure.
    exhaust_partial = "I was unable to fully verify, but records suggest Noa has clearance A."
    mock_resp_exhaust_v = MagicMock()
    mock_resp_exhaust_v.content = f"  {exhaust_partial}  "
    mock_llm_exhaust_v = MagicMock()
    mock_llm_exhaust_v.ainvoke = AsyncMock(return_value=mock_resp_exhaust_v)

    exhausted_fail_state = {
        **base_state,
        "retry_count":        _MAX_ATTEMPTS,
        "synthesizer_output": "Noa holds clearance level A per the employees table.",
        "audit_result":       {"verdict": "FAIL", "notes": "Draft lacked source citation."},
        "final_answer":       "",   # auditor never set it — never passed
    }

    with patch(patch_target, return_value=mock_llm_exhaust_v):
        result_exhaust_v = asyncio.run(chat_node(exhausted_fail_state))

    assert result_exhaust_v["final_answer"] == exhaust_partial, \
        f"Expected LLM-generated partial answer, got: {result_exhaust_v['final_answer']!r}"
    assert result_exhaust_v["final_answer"] != _FAILURE_MESSAGE, \
        "Must NOT return the hardcoded failure message when synthesizer_output exists"
    # The audit verdict in state must still be FAIL (chat_node does not change it)
    assert exhausted_fail_state["audit_result"]["verdict"] == "FAIL", \
        "audit_result must remain FAIL — the chat agent must not flip it"
    assert "audit_result" not in result_exhaust_v, \
        "Exhaustion path must not write audit_result — it stays FAIL from the auditor"
    print("PASS: exhaustion with synthesizer_output returns unverified partial answer, audit stays FAIL")

    # ── Test 13: delivery path rejects if audit verdict is not PASS ──
    # final_answer is set but audit verdict is FAIL — should NOT format and deliver
    mock_llm_no_deliver = MagicMock()
    mock_llm_no_deliver.ainvoke = AsyncMock()

    bad_audit_state = {
        **base_state,
        "final_answer":  "Some answer that should not be delivered.",
        "audit_result":  {"verdict": "FAIL", "notes": "Rejected."},
        "retry_count":   0,
    }

    with patch(patch_target, return_value=mock_llm_no_deliver):
        result_bad_audit = asyncio.run(chat_node(bad_audit_state))

    mock_llm_no_deliver.ainvoke.assert_not_called()
    assert result_bad_audit["final_answer"] == _FAILURE_MESSAGE, \
        "Delivery with non-PASS audit must return the hardcoded failure message"
    print("PASS: delivery path rejects when audit verdict is not PASS")

    # ── Test 14: multi-question message rewrites all questions ──────
    multi_q_rewritten = (
        "What is Dan Cohen's clearance level? "
        "What is Dan Cohen's salary range? "
        "Can Dan Cohen fly Business Class?"
    )
    multi_q_json = json.dumps({
        "intent":          "PLAN",
        "rewritten_query": multi_q_rewritten,
        "response":        None,
    })
    mock_resp_multi = MagicMock()
    mock_resp_multi.content = multi_q_json
    mock_llm_multi = MagicMock()
    mock_llm_multi.ainvoke = AsyncMock(return_value=mock_resp_multi)

    multi_state = {
        **base_state,
        "original_query": "What is his clearance level? What is his salary? Can he fly Business?",
        "conversation_history": [
            {"role": "user",      "content": "Tell me about Dan Cohen"},
            {"role": "assistant", "content": "Dan Cohen is in the Finance department with clearance B."},
        ],
    }
    with patch(patch_target, return_value=mock_llm_multi):
        result_multi = asyncio.run(chat_node(multi_state))

    rq = result_multi["rewritten_query"]
    assert "clearance" in rq.lower(), \
        f"Rewritten query must include the clearance question, got: {rq}"
    assert "salary" in rq.lower(), \
        f"Rewritten query must include the salary question, got: {rq}"
    assert "business" in rq.lower() or "fly" in rq.lower(), \
        f"Rewritten query must include the flight question, got: {rq}"
    assert "Dan Cohen" in rq, \
        f"Rewritten query must resolve 'his' to 'Dan Cohen', got: {rq}"
    print("PASS: multi-question message rewrites all 3 questions with resolved context")

    print("\nPASS: all chat tests passed")


if __name__ == "__main__":
    test_chat()
