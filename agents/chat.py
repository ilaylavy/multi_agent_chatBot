"""
agents/chat.py — Chat Agent node (The Face).

LangGraph node: async chat_node(state) -> dict

Handles four distinct execution paths:

  RETRY EXHAUSTION  — final_answer is empty and retry_count >= max_attempts.
                      Returns the failure message from config.yaml. No LLM call.

  CLASSIFY (initial entry) — final_answer is empty and retry_count < max_attempts.
                      Calls LLM to classify intent. Schema: { intent, response }.
                      DIRECT  → response is the answer; write to final_answer + history.
                      CLARIFY → response is a clarifying question; write to final_answer + history.
                      PLAN    → runs a dedicated rewrite LLM call to produce a
                                self-contained query, then routes to Planner.

  FORMAT AND DELIVER — final_answer is populated (set by Auditor on PASS).
                      Calls LLM to format the answer. Runs a factual-preservation
                      safety check: if the formatter introduced negation that the
                      audited draft did not contain, the formatted text is
                      discarded and the audited draft is delivered verbatim.

View    : original_query, conversation_history, final_answer, final_sources,
          chat_intent, rewritten_query
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

# The rewrite LLM gets a larger window — it must reconstruct intent across
# a full clarification cycle (original question → clarification → user detail →
# confirmation). Double the classify window by default.
MAX_REWRITE_HISTORY: int = _chat_cfg.get("max_rewrite_history", 12)


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

Post-clarification handling:
If the previous assistant message was a clarification question (it ended with
"?" and asked the user to disambiguate), treat the current user message as an
action trigger — not a new clarification need. Confirmations ("yes", "yeah",
"correct", "right"), corrections ("no, I meant X"), and added detail all
classify as PLAN. Never return CLARIFY back-to-back.

Respond with ONLY a JSON object:
{"intent": "DIRECT"|"CLARIFY"|"PLAN", "response": "your reply (required string for DIRECT/CLARIFY, null for PLAN)"}
"""

_CLASSIFY_USER_TEMPLATE = """\
{history_block}CURRENT QUERY:
{original_query}
"""


# ---------------------------------------------------------------------------
# Prompt templates — dedicated rewrite (PLAN path only)
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM_PROMPT = """\
You are a query rewriter. Your only job is to produce a single self-contained
question that a planner with zero conversation context can act on.

You must correctly handle:
  - Follow-ups ("and X?", "what about Y?") → carry forward the topic from the
    last concrete question the user asked.
  - Pronoun resolution ("his/her/their/they/them/it") → resolve to the most
    recently named entity.
  - Post-clarification confirmations — if the previous assistant turn was a
    clarification question and the user replied with "yes"/"yeah"/"correct",
    reconstruct the full intent using the clarified entity, NOT the
    clarification question itself. If they replied with a correction
    ("no, I meant X"), use the corrected entity.
  - Multi-part questions → preserve every sub-question.
  - Corrections of earlier queries → use the corrected form.

If the user's current message is already fully self-contained, copy it unchanged.
Never invent facts not present in the conversation.

Respond with ONLY a JSON object:
{"rewritten_query": "..."}
"""

_REWRITE_USER_TEMPLATE = """\
{history_block}CURRENT USER MESSAGE:
{original_query}
"""


# ---------------------------------------------------------------------------
# Prompt templates — format and deliver
# ---------------------------------------------------------------------------

_DELIVER_SYSTEM_PROMPT = """\
You are a copyeditor, not an author. Every name, number, category, date, and
conclusion in the VERIFIED ANSWER must survive in your output. You may change
tone, length, and phrasing — never facts. If the verified answer states a
positive finding (a name, a value, a rule), your output must state that same
positive finding. Never invert polarity: do not output "not found", "does not
exist", "no record", "no such X", or any negation unless the verified answer
itself contains that same negation.

Rewrite the VERIFIED ANSWER as a direct, concise response to the user's
question. Use the RECENT CONVERSATION block only to understand what the user
was asking about — never to add facts the verified answer did not contain.

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
{history_block}USER QUESTION (resolved from conversation context):
{resolved_query}

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

    rewritten_query is included so the formatter can use the self-contained
    query (set on the PLAN path) when composing the deliver prompt — without it
    the formatter sees only the raw user fragment and may misinterpret intent.
    """
    return {
        "original_query":       state["original_query"],
        "conversation_history": state["conversation_history"],
        "final_answer":         state["final_answer"],
        "final_sources":        state["final_sources"],
        "chat_intent":          state.get("chat_intent", ""),
        "rewritten_query":      state.get("rewritten_query", ""),
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


def _format_history_block(history: list, max_messages: int = MAX_CLASSIFICATION_HISTORY) -> str:
    """Return the last `max_messages` messages formatted for a chat LLM prompt.

    Default limit matches the classify call. Pass a larger value (e.g.
    MAX_REWRITE_HISTORY) for the rewrite call, which needs a broader window
    to reconstruct post-clarification intent.
    """
    recent = history[-max_messages:] if len(history) > max_messages else history
    if not recent:
        return ""
    lines = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append("")  # blank line separator
    return "\n".join(lines) + "\n"


def _format_recent_for_deliver(history: list, last_n: int = 2) -> str:
    """Return the last `last_n` messages formatted for the deliver prompt.

    Smaller window than classify/rewrite — the formatter only needs enough
    conversational context to interpret terse user fragments, not full recall.
    """
    recent = history[-last_n:] if len(history) > last_n else history
    if not recent:
        return ""
    lines = ["RECENT CONVERSATION:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append("")
    return "\n".join(lines) + "\n"


async def _rewrite_query(history: list, original_query: str, llm) -> str:
    """Produce a self-contained query for the Planner.

    Fires on the PLAN path only. Splitting the rewrite out from classify lets
    each call have a focused prompt and a broader history window, which
    matters most for post-clarification confirmations where the classify call
    would otherwise echo the clarification question itself.

    Falls back to `original_query` on any parse failure so the Planner always
    has *something* to act on.
    """
    history_block = _format_history_block(history, max_messages=MAX_REWRITE_HISTORY)
    user_message  = _REWRITE_USER_TEMPLATE.format(
        history_block=history_block,
        original_query=original_query,
    )
    try:
        response = await llm.ainvoke([
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ])
        data = parse_llm_json(response.content)
        rewritten = (data.get("rewritten_query") or "").strip()
        return rewritten or original_query
    except (ValueError, KeyError) as exc:
        logger.warning(
            "Rewrite call failed, falling back to original_query. Error: %s", exc
        )
        return original_query


# Negation phrases the formatter must never introduce on its own.
# Matched case-insensitively against both the audited draft and the
# formatted output — if the formatter added one that wasn't in the draft,
# we reject the formatted version as an unsafe polarity inversion.
_NEGATION_PATTERNS: tuple[str, ...] = (
    "not found",
    "no employee",
    "no record",
    "does not exist",
    "no such",
    "could not find",
    "no results",
    "isn't in",
    "is not in",
)


def _contains_negation(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _NEGATION_PATTERNS)


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
            intent        = data["intent"]
            response_text = data.get("response")
        except KeyError as exc:
            raise ValueError(
                f"Missing key in classify LLM output: {exc}\nRaw output: {response.content}"
            ) from exc

        # PLAN — run a dedicated rewrite call, then hand off to the Planner.
        # The rewrite is split out from classify so each prompt stays focused
        # and the rewrite gets a larger history window (critical for
        # post-clarification confirmations like "yes").
        if intent == "PLAN":
            rewritten_query = await _rewrite_query(
                view["conversation_history"],
                view["original_query"],
                llm,
            )
            logger.debug(
                "[%s] Chat classify — intent=PLAN rewritten_query=%s",
                state.get("session_id", "?"),
                rewritten_query,
            )
            updated_history = _append_messages(
                view["conversation_history"],
                view["original_query"],
            )
            return {
                "conversation_history": updated_history,
                "chat_intent":          "PLAN",
                "rewritten_query":      rewritten_query,
            }

        logger.debug(
            "[%s] Chat classify — intent=%s (no rewrite needed)",
            state.get("session_id", "?"),
            intent,
        )

        # DIRECT or CLARIFY — the response is ready immediately
        if response_text is None:
            logger.warning(
                "LLM returned intent=%r with null response — using fallback. Raw: %s",
                intent, response.content,
            )
            if intent == "DIRECT":
                response_text = "Hello! Ask me anything about your company data."
            else:  # CLARIFY
                response_text = "Could you rephrase your question?"

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

    # Feed the formatter the self-contained query (rewritten_query) plus a small
    # window of recent history. Without this, terse follow-ups like "and noa
    # levi?" reach the formatter as raw fragments, and the LLM may invert
    # polarity trying to reconcile the fragment with a concrete answer.
    resolved_query = view["rewritten_query"] or view["original_query"]
    history_block  = _format_recent_for_deliver(view["conversation_history"], last_n=2)

    user_message = _DELIVER_USER_TEMPLATE.format(
        history_block=history_block,
        resolved_query=resolved_query,
        final_answer=final_answer,
    )

    llm      = get_llm("chat")
    response = await llm.ainvoke([
        {"role": "system", "content": _DELIVER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    formatted_answer = response.content.strip()

    # Factual-preservation safety check. The audited draft (final_answer) was
    # approved by the Auditor, so falling back to it is always safe. We only
    # override when the formatter introduced a negation the draft did not have.
    if _contains_negation(formatted_answer) and not _contains_negation(final_answer):
        logger.warning(
            "[%s] Formatter introduced negation absent from audited draft — "
            "rejecting formatted output and delivering the audited draft verbatim. "
            "Draft: %r Formatted: %r",
            state.get("session_id", "?"),
            final_answer,
            formatted_answer,
        )
        formatted_answer = final_answer

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
        "original_query", "conversation_history", "final_answer", "final_sources",
        "chat_intent", "rewritten_query",
    }
    assert "plan"             not in view
    assert "task_results"     not in view
    assert "audit_result"     not in view
    assert "retry_count"      not in view
    assert "retry_notes"      not in view
    assert "synthesizer_output" not in view
    # rewritten_query is intentionally exposed so the formatter can use the
    # self-contained query; all other internal pipeline fields remain hidden.
    assert view["rewritten_query"] == CHAT_AGENT_STATE.get("rewritten_query", "")
    print("PASS: chat_view returns correct fields and excludes all forbidden ones")

    # ── Test 2: DIRECT intent — LLM responds immediately, final_answer set ──
    direct_response = "Hello! I'm a multi-agent RAG assistant. Ask me anything about your company data."
    direct_json = json.dumps({
        "intent":   "DIRECT",
        "response": direct_response,
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
    # DIRECT must only invoke the classify LLM — no rewrite call
    assert mock_llm_direct.ainvoke.call_count == 1, \
        f"DIRECT must make exactly one LLM call, got {mock_llm_direct.ainvoke.call_count}"
    print("PASS: DIRECT intent calls LLM, sets final_answer, appends both turns to history")

    # ── Test 3: CLARIFY intent — clarifying question returned as final_answer ──
    clarify_response = "Could you clarify — which Noa are you referring to?"
    clarify_json = json.dumps({
        "intent":   "CLARIFY",
        "response": clarify_response,
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
    assert mock_llm_clarify.ainvoke.call_count == 1, \
        f"CLARIFY must make exactly one LLM call, got {mock_llm_clarify.ainvoke.call_count}"
    print("PASS: CLARIFY intent calls LLM, sets final_answer to clarifying question")

    # ── Test 4: PLAN intent — classify + dedicated rewrite LLM call ──
    # PLAN now requires two sequential LLM calls: first classify returns the
    # intent, then a dedicated rewrite call produces the self-contained query.
    rewritten = "What is Dan Cohen's flight class entitlement?"
    classify_plan_json = json.dumps({"intent": "PLAN", "response": None})
    rewrite_plan_json  = json.dumps({"rewritten_query": rewritten})

    mock_resp_classify = MagicMock(); mock_resp_classify.content = classify_plan_json
    mock_resp_rewrite  = MagicMock(); mock_resp_rewrite.content  = rewrite_plan_json
    mock_llm_plan = MagicMock()
    mock_llm_plan.ainvoke = AsyncMock(side_effect=[mock_resp_classify, mock_resp_rewrite])

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
    assert mock_llm_plan.ainvoke.call_count == 2, \
        f"PLAN must make two LLM calls (classify + rewrite), got {mock_llm_plan.ainvoke.call_count}"
    print("PASS: PLAN intent fires classify + rewrite, writes rewritten_query, no final_answer")

    # ── Test 5: PLAN with empty history — rewrite returns original unchanged ──
    classify_only_json = json.dumps({"intent": "PLAN", "response": None})
    rewrite_same_json  = json.dumps({"rewritten_query": "Can Noa fly Business Class?"})
    mock_resp_c = MagicMock(); mock_resp_c.content = classify_only_json
    mock_resp_r = MagicMock(); mock_resp_r.content = rewrite_same_json
    mock_llm_no_rewrite = MagicMock()
    mock_llm_no_rewrite.ainvoke = AsyncMock(side_effect=[mock_resp_c, mock_resp_r])

    with patch(patch_target, return_value=mock_llm_no_rewrite):
        result_no_rewrite = asyncio.run(chat_node(base_state))

    assert result_no_rewrite["rewritten_query"] == base_state["original_query"]
    print("PASS: PLAN with no history passes original query through unchanged")

    # ── Test 5b: rewrite call parse failure falls back to original_query ──
    # If the rewrite LLM returns invalid JSON, the Planner must still get a
    # usable query (the original) instead of blocking the pipeline.
    classify_fallback_json = json.dumps({"intent": "PLAN", "response": None})
    mock_resp_cf = MagicMock(); mock_resp_cf.content = classify_fallback_json
    mock_resp_bad = MagicMock(); mock_resp_bad.content = "not valid json at all"
    mock_llm_fallback = MagicMock()
    mock_llm_fallback.ainvoke = AsyncMock(side_effect=[mock_resp_cf, mock_resp_bad])

    with patch(patch_target, return_value=mock_llm_fallback):
        result_fallback = asyncio.run(chat_node(base_state))

    assert result_fallback["rewritten_query"] == base_state["original_query"], \
        "Rewrite parse failure must fall back to original_query"
    print("PASS: rewrite call parse failure falls back to original_query")

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

    # Use a rewritten_query that differs from original_query so we can verify
    # the formatter receives the resolved (rewritten) form.
    deliver_state = {
        **CHAT_AGENT_STATE,
        "rewritten_query": "Can Noa fly Business Class based on her clearance level?",
        "audit_result":    {"verdict": "PASS", "notes": "All verified."},
    }
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
    assert "copyeditor" in system_msg.lower(), \
        "Delivery system prompt must cast the LLM as a copyeditor"
    assert "never invert polarity" in system_msg.lower(), \
        "Delivery system prompt must forbid polarity inversion"
    assert "remove source attribution" in system_msg.lower(), \
        "Delivery system prompt must instruct the LLM to remove source attribution"
    assert "lead with the key fact" in system_msg.lower(), \
        "Delivery system prompt must instruct the LLM to lead with the key fact"
    # User template must NOT include sources block — sources are shown separately in the UI
    user_msg = next(m["content"] for m in captured_deliver if m["role"] == "user")
    assert "SOURCES:" not in user_msg, \
        "Deliver user template must not include sources block — sources are shown in the UI"
    # The formatter must receive the rewritten (resolved) query, not the raw original
    assert "Can Noa fly Business Class based on her clearance level?" in user_msg, \
        "Deliver user template must include the rewritten_query as the resolved question"
    # Prior conversation context should reach the formatter
    assert "RECENT CONVERSATION" in user_msg, \
        "Deliver user template must include a RECENT CONVERSATION block"
    print("PASS: FORMAT AND DELIVER uses rewritten_query + history; copyeditor constraint in prompt")

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

    # ── Test 9b: classify + rewrite each receive their own truncated history ──
    # Build a state with 20 messages (10 turns). Classify should see the last 6;
    # rewrite should see the last 12. Both windows are validated independently.
    big_history = []
    for i in range(10):
        big_history.append({"role": "user",      "content": f"user-msg-{i}"})
        big_history.append({"role": "assistant",  "content": f"asst-msg-{i}"})

    classify_truncate_json = json.dumps({"intent": "PLAN", "response": None})
    rewrite_truncate_json  = json.dumps({"rewritten_query": "What is the answer?"})
    mock_resp_classify_t = MagicMock(); mock_resp_classify_t.content = classify_truncate_json
    mock_resp_rewrite_t  = MagicMock(); mock_resp_rewrite_t.content  = rewrite_truncate_json

    # Capture each call's messages separately so we can assert windows independently.
    captured_calls: list[list] = []
    responses_iter = iter([mock_resp_classify_t, mock_resp_rewrite_t])

    async def capture_each_call(messages):
        captured_calls.append(list(messages))
        return next(responses_iter)

    mock_llm_trunc = MagicMock()
    mock_llm_trunc.ainvoke = capture_each_call

    trunc_state = {
        **base_state,
        "original_query": "What is the answer?",
        "conversation_history": big_history,
    }
    with patch(patch_target, return_value=mock_llm_trunc):
        asyncio.run(chat_node(trunc_state))

    assert len(captured_calls) == 2, \
        f"PLAN must issue classify + rewrite (2 calls); got {len(captured_calls)}"

    # Call 1 (classify): only last MAX_CLASSIFICATION_HISTORY=6 messages
    classify_user_msg = next(m["content"] for m in captured_calls[0] if m["role"] == "user")
    assert "user-msg-0" not in classify_user_msg, \
        "Classify must not send old history (user-msg-0)"
    assert "user-msg-6" not in classify_user_msg, \
        "Classify must not send user-msg-6 (outside last 6)"
    assert "user-msg-7" in classify_user_msg, "Classify must include user-msg-7"
    assert "asst-msg-9" in classify_user_msg, "Classify must include most recent asst-msg-9"

    # Call 2 (rewrite): last MAX_REWRITE_HISTORY=12 messages — wider window
    rewrite_user_msg = next(m["content"] for m in captured_calls[1] if m["role"] == "user")
    assert "user-msg-0" not in rewrite_user_msg, \
        "Rewrite must not send oldest history (user-msg-0) when 20 messages exist"
    # Rewrite window is 12 → should include user-msg-4 (message index 8, within last 12)
    assert "user-msg-4" in rewrite_user_msg, \
        "Rewrite must include user-msg-4 (inside last 12 messages)"
    assert "asst-msg-9" in rewrite_user_msg, \
        "Rewrite must include most recent asst-msg-9"

    assert len(trunc_state["conversation_history"]) == 20, \
        "Full conversation_history must not be mutated"
    print(
        f"PASS: classify receives last {MAX_CLASSIFICATION_HISTORY} messages, "
        f"rewrite receives last {MAX_REWRITE_HISTORY} messages"
    )

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
    # Classify returns PLAN; the dedicated rewrite call produces the full
    # 3-part rewritten query with pronouns resolved.
    multi_q_rewritten = (
        "What is Dan Cohen's clearance level? "
        "What is Dan Cohen's salary range? "
        "Can Dan Cohen fly Business Class?"
    )
    mock_resp_multi_classify = MagicMock()
    mock_resp_multi_classify.content = json.dumps({"intent": "PLAN", "response": None})
    mock_resp_multi_rewrite  = MagicMock()
    mock_resp_multi_rewrite.content  = json.dumps({"rewritten_query": multi_q_rewritten})
    mock_llm_multi = MagicMock()
    mock_llm_multi.ainvoke = AsyncMock(
        side_effect=[mock_resp_multi_classify, mock_resp_multi_rewrite]
    )

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

    # ── Test 15: classify prompt contains post-clarification PLAN guidance ──
    # The classifier must route confirmations after a clarification question
    # to PLAN, not loop back to CLARIFY. We verify two things: (a) the prompt
    # itself contains the post-clarification rule, and (b) when the mock
    # returns PLAN, the path is taken as PLAN (not re-clarified).
    assert "post-clarification" in _CLASSIFY_SYSTEM_PROMPT.lower(), \
        "Classify prompt must mention post-clarification handling"
    assert "never return clarify back-to-back" in _CLASSIFY_SYSTEM_PROMPT.lower(), \
        "Classify prompt must forbid back-to-back CLARIFY responses"

    post_clar_classify_json = json.dumps({"intent": "PLAN", "response": None})
    post_clar_rewrite_json  = json.dumps({
        "rewritten_query": "What is the salary range for Noa Levy?"
    })
    mock_resp_pc_c = MagicMock(); mock_resp_pc_c.content = post_clar_classify_json
    mock_resp_pc_r = MagicMock(); mock_resp_pc_r.content = post_clar_rewrite_json
    mock_llm_post_clar = MagicMock()
    mock_llm_post_clar.ainvoke = AsyncMock(side_effect=[mock_resp_pc_c, mock_resp_pc_r])

    post_clar_state = {
        **base_state,
        "original_query": "yes",
        "conversation_history": [
            {"role": "user",      "content": "maybe noa levy"},
            {"role": "assistant", "content": "Did you mean Noa Levy (not Levi)?"},
        ],
    }
    with patch(patch_target, return_value=mock_llm_post_clar):
        result_pc = asyncio.run(chat_node(post_clar_state))

    assert result_pc["chat_intent"] == "PLAN", \
        f"'yes' after a clarification must route as PLAN, got {result_pc['chat_intent']!r}"
    assert "Noa Levy" in result_pc["rewritten_query"], \
        f"Rewritten query must name Noa Levy, got: {result_pc['rewritten_query']!r}"
    print("PASS: post-clarification 'yes' routes as PLAN, rewrite reconstructs intent")

    # ── Test 16: rewrite call sees enough history to reconstruct intent ──
    # Build a state where the last assistant turn is a clarification and the
    # user replied "yes". Capture the rewrite call's user message and assert
    # it contains both the clarification question and the user's earlier
    # concrete question about salary.
    captured_rewrite: list = []
    rewrite_capture_json = json.dumps({
        "rewritten_query": "What is the salary range for Noa Levy?"
    })
    mock_resp_rw_classify = MagicMock()
    mock_resp_rw_classify.content = json.dumps({"intent": "PLAN", "response": None})
    mock_resp_rw = MagicMock(); mock_resp_rw.content = rewrite_capture_json

    _rw_iter = iter([mock_resp_rw_classify, mock_resp_rw])
    async def capture_rewrite_call(messages):
        captured_rewrite.append(list(messages))
        return next(_rw_iter)

    mock_llm_rewrite = MagicMock()
    mock_llm_rewrite.ainvoke = capture_rewrite_call

    rewrite_state = {
        **base_state,
        "original_query": "yes",
        "conversation_history": [
            {"role": "user",      "content": "What is the salary range for Noa Levi?"},
            {"role": "assistant", "content": "I could not find an employee named Noa Levi. Did you mean a similar name?"},
            {"role": "user",      "content": "maybe noa levy"},
            {"role": "assistant", "content": "Did you mean Noa Levy (not Levi)?"},
        ],
    }
    with patch(patch_target, return_value=mock_llm_rewrite):
        result_rw = asyncio.run(chat_node(rewrite_state))

    assert len(captured_rewrite) == 2, \
        f"Expected classify + rewrite (2 calls), got {len(captured_rewrite)}"

    # Inspect the rewrite call's user message (second call)
    rewrite_user_msg   = next(m["content"] for m in captured_rewrite[1] if m["role"] == "user")
    rewrite_system_msg = next(m["content"] for m in captured_rewrite[1] if m["role"] == "system")

    assert "salary range for Noa Levi" in rewrite_user_msg, \
        "Rewrite call must see the user's original salary question in history"
    assert "Did you mean Noa Levy" in rewrite_user_msg, \
        "Rewrite call must see the clarification question in history"
    assert "CURRENT USER MESSAGE" in rewrite_user_msg and "yes" in rewrite_user_msg, \
        "Rewrite call must include the current 'yes' message"
    # The rewrite system prompt must teach the post-clarification behavior
    assert "post-clarification" in rewrite_system_msg.lower(), \
        "Rewrite system prompt must teach post-clarification reconstruction"

    assert "Noa Levy" in result_rw["rewritten_query"], \
        f"Rewritten query must contain 'Noa Levy', got: {result_rw['rewritten_query']!r}"
    assert "salary" in result_rw["rewritten_query"].lower(), \
        f"Rewritten query must preserve the salary topic, got: {result_rw['rewritten_query']!r}"
    print("PASS: rewrite call reconstructs intent from clarification confirmation with history")

    # ── Test 17: formatter factual-preservation safety check ──
    # (a) formatter inverts polarity — audited draft has no negation but
    #     formatted output does → safety check must override with draft.
    audited_positive_draft = (
        "Noa Levi's salary is in the $85,000 - $110,000 range per year, "
        "based on the employees table."
    )
    inverted_formatted = "There is no employee named Noa Levi in the system."
    mock_resp_invert = MagicMock(); mock_resp_invert.content = inverted_formatted
    mock_llm_invert = MagicMock()
    mock_llm_invert.ainvoke = AsyncMock(return_value=mock_resp_invert)

    invert_state = {
        **base_state,
        "final_answer":    audited_positive_draft,
        "final_sources":   [{"source_id": "employees", "source_type": "csv", "label": "Employees"}],
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
        "rewritten_query": "What is the salary range for Noa Levi?",
    }
    with patch(patch_target, return_value=mock_llm_invert):
        result_invert = asyncio.run(chat_node(invert_state))

    assert result_invert["final_answer"] == audited_positive_draft, \
        f"Safety check must revert to the audited draft when the formatter "\
        f"introduces negation. Got: {result_invert['final_answer']!r}"
    # History must record the audited draft, not the inverted formatted output
    assert result_invert["conversation_history"][-1]["content"] == audited_positive_draft
    print("PASS: formatter polarity inversion triggers safety check, audited draft delivered")

    # (b) both draft and formatted contain legitimate negation → accept formatted
    negative_draft = "No employee named Noa Levie was found in the records."
    negative_formatted = "There is no employee named Noa Levie."
    mock_resp_neg = MagicMock(); mock_resp_neg.content = negative_formatted
    mock_llm_neg = MagicMock()
    mock_llm_neg.ainvoke = AsyncMock(return_value=mock_resp_neg)

    neg_state = {
        **base_state,
        "final_answer":    negative_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
        "rewritten_query": "What is the salary range for Noa Levie?",
    }
    with patch(patch_target, return_value=mock_llm_neg):
        result_neg = asyncio.run(chat_node(neg_state))

    assert result_neg["final_answer"] == negative_formatted, \
        "When the audited draft already contains negation, the formatter's "\
        "negated output is legitimate and must be delivered."
    print("PASS: formatter legitimate negation (matches audited draft) is delivered unchanged")

    # (c) neither draft nor formatted contains negation → formatted delivered
    plain_draft     = "Dan Cohen earns $95,000 annually based on the employees table."
    plain_formatted = "Dan Cohen makes $95,000 per year."
    mock_resp_plain = MagicMock(); mock_resp_plain.content = plain_formatted
    mock_llm_plain = MagicMock()
    mock_llm_plain.ainvoke = AsyncMock(return_value=mock_resp_plain)

    plain_state = {
        **base_state,
        "final_answer":    plain_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
        "rewritten_query": "What is Dan Cohen's salary?",
    }
    with patch(patch_target, return_value=mock_llm_plain):
        result_plain = asyncio.run(chat_node(plain_state))

    assert result_plain["final_answer"] == plain_formatted, \
        "When neither draft nor formatted has negation, formatted must be delivered."
    print("PASS: formatter non-negation rewrite is delivered unchanged")

    # ── Test 18: 4-turn regression from the original trace ──
    # Simulates: Dan Cohen salary → follow-up "and noa levi?" → ambiguous
    # "maybe noa levy" → confirmation "yes". The critical assertion is that
    # turn 4's rewritten_query names Noa Levy and salary, and is NOT the
    # clarification question echoed back.
    class _Seq:
        def __init__(self, responses):
            self._it = iter(responses)
        async def ainvoke(self, messages):
            resp = MagicMock()
            resp.content = next(self._it)
            return resp

    # Turn 1 — "What is the salary range for Dan Cohen?"
    t1_rewritten = "What is the salary range for Dan Cohen?"
    turn1_mock = _Seq([
        json.dumps({"intent": "PLAN", "response": None}),                # classify
        json.dumps({"rewritten_query": t1_rewritten}),                   # rewrite
    ])
    turn1_state = {
        **base_state,
        "original_query":       "What is the salary range for Dan Cohen?",
        "conversation_history": [],
    }
    with patch(patch_target, return_value=turn1_mock):
        r1 = asyncio.run(chat_node(turn1_state))
    assert r1["chat_intent"] == "PLAN"
    assert "Dan Cohen" in r1["rewritten_query"]

    # Turn 2 — "and noa levi?" — must rewrite to the full salary question
    t2_rewritten = "What is the salary range for Noa Levi?"
    turn2_mock = _Seq([
        json.dumps({"intent": "PLAN", "response": None}),
        json.dumps({"rewritten_query": t2_rewritten}),
    ])
    turn2_history = r1["conversation_history"] + [
        {"role": "assistant", "content": "Dan Cohen's salary range is $90,000 - $110,000."},
    ]
    turn2_state = {
        **base_state,
        "original_query":       "and noa levi?",
        "conversation_history": turn2_history,
    }
    with patch(patch_target, return_value=turn2_mock):
        r2 = asyncio.run(chat_node(turn2_state))
    assert r2["chat_intent"] == "PLAN"
    assert "Noa Levi" in r2["rewritten_query"], \
        f"Turn 2 must rewrite 'and noa levi?' into a full question about Noa Levi, "\
        f"got: {r2['rewritten_query']!r}"
    assert "salary" in r2["rewritten_query"].lower()

    # Turn 3 — "maybe noa levy" — CLARIFY is acceptable (ambiguous)
    t3_clar = "Did you mean Noa Levy (not Levi)?"
    turn3_mock = _Seq([
        json.dumps({"intent": "CLARIFY", "response": t3_clar}),
    ])
    turn3_history = r2["conversation_history"] + [
        {"role": "assistant", "content": "I could not find an employee named Noa Levi."},
    ]
    turn3_state = {
        **base_state,
        "original_query":       "maybe noa levy",
        "conversation_history": turn3_history,
    }
    with patch(patch_target, return_value=turn3_mock):
        r3 = asyncio.run(chat_node(turn3_state))
    assert r3["chat_intent"] in ("CLARIFY", "PLAN"), \
        f"Turn 3 may classify as CLARIFY or PLAN, got {r3['chat_intent']!r}"

    # Turn 4 — "yes" — MUST route as PLAN and reconstruct "salary range for Noa Levy"
    # NEVER echo the clarification question itself as the rewritten query.
    t4_rewritten = "What is the salary range for Noa Levy?"
    turn4_mock = _Seq([
        json.dumps({"intent": "PLAN", "response": None}),
        json.dumps({"rewritten_query": t4_rewritten}),
    ])
    turn4_history = r3["conversation_history"]   # already contains the clarification
    turn4_state = {
        **base_state,
        "original_query":       "yes",
        "conversation_history": turn4_history,
    }
    with patch(patch_target, return_value=turn4_mock):
        r4 = asyncio.run(chat_node(turn4_state))
    assert r4["chat_intent"] == "PLAN", \
        f"Turn 4 'yes' must route as PLAN, got {r4['chat_intent']!r}"
    rq4 = r4["rewritten_query"]
    assert "Noa Levy" in rq4, \
        f"Turn 4 rewritten query must name 'Noa Levy', got: {rq4!r}"
    assert "salary" in rq4.lower(), \
        f"Turn 4 rewritten query must preserve the salary topic, got: {rq4!r}"
    assert rq4 != t3_clar, \
        f"Turn 4 rewrite must NOT echo the clarification question, got: {rq4!r}"
    print("PASS: 4-turn regression — turn-4 'yes' reconstructs full Noa Levy salary intent")

    print("\nPASS: all chat tests passed")


if __name__ == "__main__":
    test_chat()
