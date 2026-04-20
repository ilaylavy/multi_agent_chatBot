"""
agents/chat.py — Chat Agent node (The Face).

LangGraph node: async chat_node(state) -> dict

Four execution paths:

  RETRY EXHAUSTION  — final_answer empty + retry_count >= max_attempts.
                      Returns a failure message (LLM-composed partial answer
                      when synthesizer_output exists, else the config default).

  FAST GREETING     — obvious greetings match a regex; no LLM call.
                      Returns a canned DIRECT response.

  REASON AND ROUTE  — single reasoning LLM call with chain of thought
                      (understand → assess → act). Extracts stable user
                      facts into session_context ambiently, independent of
                      the decision. Produces one of DIRECT, CLARIFY, or
                      PLAN with a self-contained rewritten query.

  FORMAT AND DELIVER — final_answer populated by pipeline (audit verdict
                      PASS). LLM reformats the verified answer. Post-format
                      safety check rejects polarity inversions.

View    : original_query, conversation_history, final_answer, final_sources,
          chat_intent, rewritten_query
Returns : { conversation_history, chat_intent, rewritten_query, chat_reasoning }          on PLAN
          { conversation_history, chat_intent, final_answer, chat_reasoning }              on DIRECT / CLARIFY
          { conversation_history, final_answer }                                            on FORMAT or EXHAUSTION
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from core.data_context import get_data_context
from core.llm_config import _load_config, get_llm
from core.parse import parse_llm_json
from core.session_context import get_session_context, update_session_context
from core.state import AgentState, Message

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_retry_cfg       = _load_config()["retry"]
_FAILURE_MESSAGE: str = _retry_cfg["failure_message"]
_MAX_ATTEMPTS:    int = _retry_cfg["max_attempts"]

# Reasoning call gets a generous history window — it must reconstruct intent
# across clarification cycles and keep session facts consistent. Configurable
# via config.yaml chat.max_reasoning_history; default 12.
_chat_cfg = _load_config().get("chat", {})
MAX_REASONING_HISTORY: int = _chat_cfg.get("max_reasoning_history", 12)


# ---------------------------------------------------------------------------
# Prompt templates — single reasoning call
# ---------------------------------------------------------------------------

_REASONING_SYSTEM_PROMPT = """\
You are the entry agent of a data-answering system. You do one thing per turn:
reason about the user's message and produce a JSON decision that either
answers directly, asks for clarification, or hands a self-contained query to
the downstream pipeline.

Think step by step in three stages. Surface the result as structured JSON —
not prose — exactly matching the schema at the end.

STAGE 1 — UNDERSTAND
  - State the user's intent in one sentence.
  - Scan the user's message for self-identifying facts. If any are present, you need to record them in
    session_context_update using stable keys. Extraction is ambient — do it on every turn,
    independent of your final decision. Corrections overwrite prior values.

STAGE 2 — ASSESS
  - Decide whether the user is asking for facts from the system's data, or
    doing something else (chatting, greeting, giving a self-identifying fact,
    providing information without asking a question, or asking something
    outside the DATA CONTEXT).
  - If the user is asking for data, identify gaps that would block a
    self-contained rewrite:
      * unresolved self-reference ("my", "I", "mine") not covered by
        SESSION CONTEXT
      * unresolved pronoun ("he", "she", "they", "it") with no clear
        referent in recent history
      * ambiguous entity critical to the query
    Never judge whether the system *can* answer the question — that is the
    downstream planner's job. Only flag gaps that prevent writing a single
    self-contained sentence.

STAGE 3 — ACT
  - DIRECT   — anything other than a data lookup: greetings, identity
               statements, out-of-scope questions, providing information
               without asking a data question, conversational responses.
               Write direct_response. Leave rewritten_query and
               clarifying_question empty.
  - CLARIFY  — a data question with unresolvable gaps. Write ONE short
               clarifying_question. Do not chain clarifications — if the
               previous turn was a clarification and the user has now
               answered, move forward rather than asking again.
  - PLAN     — a data question with no unresolved gaps. Write rewritten_query
               as ONE self-contained question a person would naturally ask,
               including every entity and constraint needed to answer it.
               Do NOT reference sources, tables, records, files, or
               documents. Do NOT use procedural language such as "retrieve",
               "provide", "lookup", "if available", or "otherwise indicate".
               Phrase it as a question a user would type, not an instruction
               to a system. The downstream planner sees no conversation
               history, so the question must stand alone.

Rules that apply across stages:
  - Post-clarification confirmations ("yes", "right", "correct") must
    reconstruct the original concrete question using the clarified entity,
    NEVER echo the clarification itself as the rewritten_query.
  - DATA CONTEXT is for out-of-scope detection only — never list data
    sources to the user, and never guess whether an answer exists.

Respond with ONLY a JSON object in this exact shape. All fields must be
present; use empty strings or empty arrays/objects where not applicable:
{
  "reasoning": {
    "user_intent":         "one sentence",
    "gaps":                [],
    "decision_rationale":  "one sentence"
  },
  "decision":                "PLAN|CLARIFY|DIRECT",
  "rewritten_query":         "",
  "clarifying_question":     "",
  "direct_response":         "",
  "session_context_update":  {}
}
"""

_REASONING_USER_TEMPLATE = """\
{data_context_block}{session_context_block}{history_block}CURRENT USER MESSAGE:
{original_query}
"""


# ---------------------------------------------------------------------------
# Prompt templates — format and deliver (unchanged)
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
    recommendations not directly stated in the answer.
  - Never refer to source documents by name in the formatted answer.
  - Do not reveal internal system details.
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
    Returns only the fields the Chat Agent's LLM prompts may see.
    Never exposes plan, task_results, audit_result, retry_count, retry_notes,
    or chat_reasoning (written by this agent — must not feed back into its
    own prompt).

    rewritten_query is included so the formatter can use the self-contained
    query (set on the PLAN path) when composing the deliver prompt.
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
# Internal helpers — prompt formatting
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


def _format_history_block(history: list, max_messages: int = MAX_REASONING_HISTORY) -> str:
    """Return the last `max_messages` messages formatted for the reasoning prompt."""
    recent = history[-max_messages:] if len(history) > max_messages else history
    if not recent:
        return ""
    lines = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_recent_for_deliver(history: list, last_n: int = 2) -> str:
    """Return the last `last_n` messages formatted for the deliver prompt."""
    recent = history[-last_n:] if len(history) > last_n else history
    if not recent:
        return ""
    lines = ["RECENT CONVERSATION:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_session_context_block(ctx: dict) -> str:
    """Render the session context dict for the reasoning prompt."""
    if not ctx:
        return "SESSION CONTEXT: (empty — nothing known about this user yet)\n\n"
    lines = ["SESSION CONTEXT (stable facts from prior turns):"]
    for k, v in ctx.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_data_context_block(data_context: str) -> str:
    if not data_context:
        return ""
    return f"DATA CONTEXT:\n{data_context}\n\n"


# Negation phrases the formatter must never introduce on its own.
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
    if not updated or updated[-1].get("content") != query:
        updated.append(Message(role="user", content=query))
    if answer is not None:
        updated.append(Message(role="assistant", content=answer))
    return updated


# ---------------------------------------------------------------------------
# Reasoning output helpers
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"PLAN", "CLARIFY", "DIRECT"}


def _fallback_reasoning(original_query: str, raw: str | None = None) -> dict:
    """Return a safe reasoning payload when parsing fails.

    The Planner always gets *something*, so a malformed LLM response cannot
    block the pipeline entirely — it degrades to a straight PLAN with the
    user's original query.
    """
    if raw:
        logger.warning("Reasoning call produced unparseable output; falling back to PLAN. Raw: %s", raw[:400])
    return {
        "reasoning": {
            "user_intent":         "fallback — reasoning call failed to return parseable JSON",
            "gaps":                [],
            "decision_rationale":  "Parse failure — default to PLAN with original query.",
        },
        "decision":               "PLAN",
        "rewritten_query":        original_query,
        "clarifying_question":    "",
        "direct_response":        "",
        "session_context_update": {},
    }


def _normalize_reasoning(data: dict, original_query: str) -> dict:
    """Ensure the reasoning dict has every expected field with safe defaults."""
    reasoning = data.get("reasoning") or {}
    decision  = (data.get("decision") or "").upper().strip()
    if decision not in _VALID_DECISIONS:
        logger.warning("Reasoning returned invalid decision %r — coercing to PLAN", decision)
        decision = "PLAN"

    rewritten = (data.get("rewritten_query") or "").strip()
    if decision == "PLAN" and not rewritten:
        rewritten = original_query

    return {
        "reasoning": {
            "user_intent":         reasoning.get("user_intent", "") or "",
            "gaps":                reasoning.get("gaps", []) or [],
            "decision_rationale":  reasoning.get("decision_rationale", "") or "",
        },
        "decision":               decision,
        "rewritten_query":        rewritten,
        "clarifying_question":   (data.get("clarifying_question") or "").strip(),
        "direct_response":       (data.get("direct_response") or "").strip(),
        "session_context_update": data.get("session_context_update") or {},
    }


def _serialize_reasoning(payload: dict) -> str:
    """Render the reasoning dict as a stable JSON string for the trace."""
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Fast pre-classification — skip LLM for obvious greetings
# ---------------------------------------------------------------------------

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
    retry_count  = state["retry_count"]
    session_id   = state.get("session_id", "?")

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

    # ── Path 2: REASON AND ROUTE ──────────────────────────────────
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

        # Build the three context blocks the reasoning prompt expects
        data_context_block    = _format_data_context_block(get_data_context())
        session_context_block = _format_session_context_block(get_session_context(session_id))
        history_block         = _format_history_block(view["conversation_history"])

        user_message = _REASONING_USER_TEMPLATE.format(
            data_context_block=data_context_block,
            session_context_block=session_context_block,
            history_block=history_block,
            original_query=view["original_query"],
        )

        llm      = get_llm("chat")
        response = await llm.ainvoke([
            {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ])

        try:
            data = parse_llm_json(response.content)
            payload = _normalize_reasoning(data, view["original_query"])
        except ValueError as exc:
            logger.warning("[%s] reasoning parse failed: %s", session_id, exc)
            payload = _fallback_reasoning(view["original_query"], raw=str(response.content))

        # Ambient: apply session_context_update on every path, independent of decision.
        context_update = payload.get("session_context_update") or {}
        if context_update:
            update_session_context(session_id, context_update)

        chat_reasoning_json = _serialize_reasoning(payload)
        decision = payload["decision"]

        logger.debug(
            "[%s] Chat reasoning — decision=%s update_keys=%s",
            session_id, decision, list(context_update.keys()),
        )

        if decision == "PLAN":
            rewritten = payload["rewritten_query"] or view["original_query"]
            updated_history = _append_messages(
                view["conversation_history"],
                view["original_query"],
            )
            return {
                "conversation_history": updated_history,
                "chat_intent":          "PLAN",
                "rewritten_query":      rewritten,
                "chat_reasoning":       chat_reasoning_json,
            }

        if decision == "CLARIFY":
            response_text = payload["clarifying_question"] or "Could you rephrase your question?"
        else:  # DIRECT
            response_text = payload["direct_response"] or "Hello! Ask me anything about your company data."

        updated_history = _append_messages(
            view["conversation_history"],
            view["original_query"],
            response_text,
        )
        return {
            "conversation_history": updated_history,
            "chat_intent":          decision,
            "final_answer":         response_text,
            "chat_reasoning":       chat_reasoning_json,
        }

    # ── Path 3: FORMAT AND DELIVER ────────────────────────────────
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
    # window of recent history. Without this, terse follow-ups reach the
    # formatter as raw fragments and the LLM may invert polarity trying to
    # reconcile the fragment with a concrete answer.
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

    # Factual-preservation safety check — reject formatter polarity inversions.
    if _contains_negation(formatted_answer) and not _contains_negation(final_answer):
        logger.warning(
            "[%s] Formatter introduced negation absent from audited draft — "
            "rejecting formatted output and delivering the audited draft verbatim. "
            "Draft: %r Formatted: %r",
            session_id,
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
    from unittest.mock import AsyncMock, MagicMock, patch

    from core.session_context import clear_session_context
    from tests.fixtures import CHAT_AGENT_STATE

    patch_target        = f"{__name__}.get_llm"
    data_ctx_patch      = f"{__name__}.get_data_context"
    fake_data_context   = "The system contains the following data:\n- Example Source (record): description"

    # ── Shared minimal state skeleton ────────────────────────────
    base_state: AgentState = {
        "original_query":       "Can Noa fly Business Class?",
        "session_id":           "test-session-001",
        "conversation_history": [],
        "chat_intent":          "",
        "rewritten_query":      "",
        "chat_reasoning":       "",
        "plan":                 [],
        "manifest_context":     "",
        "planner_reasoning":    "",
        "task_results":         {},
        "sources_used":         [],
        "retrieved_chunks":     [],
        "draft_answer":         "",
        "synthesizer_output":   "",
        "audit_result":         {"verdict": "PASS", "notes": ""},
        "retry_count":          0,
        "retry_notes":          "",
        "retry_history":        [],
        "final_answer":         "",
        "final_sources":        [],
    }

    def _reasoning_payload(
        decision,
        rewritten_query="",
        direct_response="",
        clarifying_question="",
        session_context_update=None,
        user_intent="",
        gaps=None,
    ):
        import json as _json
        return _json.dumps({
            "reasoning": {
                "user_intent":        user_intent or f"user requested {decision.lower()}",
                "gaps":               gaps or [],
                "decision_rationale": f"selected {decision}",
            },
            "decision":               decision,
            "rewritten_query":        rewritten_query,
            "clarifying_question":    clarifying_question,
            "direct_response":        direct_response,
            "session_context_update": session_context_update or {},
        })

    def _mock_llm(json_str: str) -> MagicMock:
        resp = MagicMock()
        resp.content = json_str
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=resp)
        return llm

    # Every reasoning-path test runs with data_context mocked — otherwise the
    # module tries to load the real manifest on disk.
    def _run(state: dict, llm, sid: str = "test-session-001") -> dict:
        state = {**state, "session_id": sid}
        with patch(data_ctx_patch, return_value=fake_data_context), \
             patch(patch_target, return_value=llm):
            return asyncio.run(chat_node(state))

    # Ensure a clean session_context between tests
    def _reset_sid(sid: str = "test-session-001") -> None:
        clear_session_context(sid)

    # ── Test 1: chat_view exposes only the permitted fields ───────
    view = chat_view(CHAT_AGENT_STATE)
    assert set(view.keys()) == {
        "original_query", "conversation_history", "final_answer", "final_sources",
        "chat_intent", "rewritten_query",
    }
    for forbidden in ("plan", "task_results", "audit_result", "retry_count",
                      "retry_notes", "synthesizer_output", "chat_reasoning"):
        assert forbidden not in view, f"chat_view leaked {forbidden}"
    print("PASS: chat_view exposes only permitted fields; hides chat_reasoning from its own prompt")

    # ── Test 2: DIRECT — LLM answers immediately, no pipeline ─────
    _reset_sid()
    direct_response = "I can help with questions about your company data. What would you like to know?"
    direct_json = _reasoning_payload("DIRECT", direct_response=direct_response)

    direct_state = {**base_state, "original_query": "What can you do?"}
    result_direct = _run(direct_state, _mock_llm(direct_json))

    assert result_direct["chat_intent"]   == "DIRECT"
    assert result_direct["final_answer"]  == direct_response
    assert "chat_reasoning" in result_direct, "DIRECT must write chat_reasoning"
    assert "rewritten_query" not in result_direct, "DIRECT must not write rewritten_query"
    hist = result_direct["conversation_history"]
    assert hist[-1] == {"role": "assistant", "content": direct_response}
    print("PASS: DIRECT writes final_answer + chat_reasoning; no rewritten_query")

    # ── Test 3: CLARIFY ───────────────────────────────────────────
    _reset_sid()
    clar_question = "Which Noa do you mean — Noa Levi or Noa Levy?"
    clar_json = _reasoning_payload(
        "CLARIFY",
        clarifying_question=clar_question,
        gaps=["ambiguous entity 'Noa'"],
    )
    result_clar = _run({**base_state, "original_query": "Can Noa fly Business Class?"},
                       _mock_llm(clar_json))
    assert result_clar["chat_intent"]  == "CLARIFY"
    assert result_clar["final_answer"] == clar_question
    assert "chat_reasoning" in result_clar
    print("PASS: CLARIFY returns clarifying_question as final_answer")

    # ── Test 4: PLAN — sets rewritten_query, no final_answer ──────
    _reset_sid()
    rewritten = "What is Dan Cohen's flight class entitlement?"
    plan_json = _reasoning_payload(
        "PLAN",
        rewritten_query=rewritten,
    )
    plan_state = {
        **base_state,
        "original_query": "What about Dan?",
        "conversation_history": [
            {"role": "user",      "content": "Can Noa fly Business Class?"},
            {"role": "assistant", "content": "Yes, Noa holds clearance level A."},
        ],
    }
    result_plan = _run(plan_state, _mock_llm(plan_json))
    assert result_plan["chat_intent"]     == "PLAN"
    assert result_plan["rewritten_query"] == rewritten
    assert "chat_reasoning" in result_plan
    assert "final_answer" not in result_plan, "PLAN must not set final_answer"
    print("PASS: PLAN writes rewritten_query + chat_reasoning, leaves final_answer empty")

    # ── Test 5: reasoning parse failure falls back to PLAN ────────
    _reset_sid()
    bad_resp = MagicMock(); bad_resp.content = "not valid json at all"
    bad_llm  = MagicMock(); bad_llm.ainvoke = AsyncMock(return_value=bad_resp)
    result_bad = _run(base_state, bad_llm)
    assert result_bad["chat_intent"]     == "PLAN"
    assert result_bad["rewritten_query"] == base_state["original_query"]
    assert "chat_reasoning" in result_bad
    print("PASS: reasoning parse failure falls back to PLAN with original_query")

    # ── Test 6: session_context_update persists across turns ──────
    _reset_sid()
    id_json = _reasoning_payload(
        "DIRECT",
        direct_response="Nice to meet you, Dan. Ask me anything about your data.",
        session_context_update={"user_name": "Dan", "department": "Engineering"},
    )
    _run({**base_state, "original_query": "my name is Dan, I'm in Engineering"}, _mock_llm(id_json))
    ctx_after = get_session_context("test-session-001")
    assert ctx_after.get("user_name")  == "Dan"
    assert ctx_after.get("department") == "Engineering"
    print("PASS: identity-only message persists user_name + department in session_context")

    # ── Test 6b: session_context correction overwrites on next turn ──
    correction_json = _reasoning_payload(
        "DIRECT",
        direct_response="Got it, Noa.",
        session_context_update={"user_name": "Noa"},
    )
    _run({**base_state, "original_query": "actually I'm Noa, not Dan"}, _mock_llm(correction_json))
    ctx_corrected = get_session_context("test-session-001")
    assert ctx_corrected.get("user_name")  == "Noa",         "correction must overwrite user_name"
    assert ctx_corrected.get("department") == "Engineering", "unchanged keys must survive"
    print("PASS: session_context correction overwrites the collision key")

    # ── Test 7a: retry exhaustion, no synthesizer → failure message, no LLM ──
    _reset_sid()
    mock_llm_exhaust = MagicMock(); mock_llm_exhaust.ainvoke = AsyncMock()
    exh_state = {**base_state, "retry_count": _MAX_ATTEMPTS, "synthesizer_output": ""}
    result_exh = _run(exh_state, mock_llm_exhaust)
    mock_llm_exhaust.ainvoke.assert_not_called()
    assert result_exh["final_answer"] == _FAILURE_MESSAGE
    print("PASS: exhaustion without synthesizer_output returns _FAILURE_MESSAGE, no LLM call")

    # ── Test 7b: retry exhaustion with synthesizer_output → LLM composes partial ──
    _reset_sid()
    partial = "I could not fully verify, but records suggest Noa has clearance A."
    partial_resp = MagicMock(); partial_resp.content = f"  {partial}  "
    partial_llm  = MagicMock(); partial_llm.ainvoke = AsyncMock(return_value=partial_resp)
    exh_state2 = {
        **base_state,
        "retry_count":        _MAX_ATTEMPTS,
        "synthesizer_output": "Noa holds clearance level A per the employees table.",
    }
    result_exh2 = _run(exh_state2, partial_llm)
    assert result_exh2["final_answer"] == partial
    print("PASS: exhaustion with synthesizer_output calls LLM and returns partial answer")

    # ── Test 8: fast greeting — no LLM call ───────────────────────
    _reset_sid()
    fast_llm = MagicMock(); fast_llm.ainvoke = AsyncMock()
    for greeting in ("hi", "hello", "Hey!", "thanks", "bye"):
        result_fast = _run({**base_state, "original_query": greeting}, fast_llm)
        assert result_fast["chat_intent"]  == "DIRECT"
        assert result_fast["final_answer"]
        assert "chat_reasoning" not in result_fast, "fast path should not emit reasoning"
    fast_llm.ainvoke.assert_not_called()
    print("PASS: fast greeting path returns DIRECT without any LLM call")

    # ── Test 9: delivery path ─ audit PASS ─── uses rewritten_query + recent history ──
    _reset_sid()
    audited_draft = "Yes. Noa holds clearance level A, which entitles her to Business Class on flights over 4 hours."
    formatted     = "Noa has clearance level A and is entitled to Business Class on flights over 4 hours."
    deliver_resp  = MagicMock(); deliver_resp.content = f"  {formatted}  "
    captured: list = []
    async def capture(messages):
        captured.extend(messages)
        return deliver_resp
    deliver_llm = MagicMock(); deliver_llm.ainvoke = capture

    deliver_state = {
        **base_state,
        "original_query":   "Can she fly Business?",
        "rewritten_query":  "Can Noa fly Business Class?",
        "conversation_history": [
            {"role": "user",      "content": "Who is Noa?"},
            {"role": "assistant", "content": "Noa is a senior engineer."},
        ],
        "final_answer":    audited_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
    }
    result_deliver = _run(deliver_state, deliver_llm)
    assert set(result_deliver.keys()) == {"conversation_history", "final_answer"}
    assert result_deliver["final_answer"] == formatted
    system_msg = next(m["content"] for m in captured if m["role"] == "system")
    user_msg   = next(m["content"] for m in captured if m["role"] == "user")
    assert "copyeditor" in system_msg.lower()
    assert "never invert polarity" in system_msg.lower()
    # Formatter must use rewritten_query, not original_query
    assert "Can Noa fly Business Class?" in user_msg
    assert "RECENT CONVERSATION" in user_msg
    print("PASS: delivery uses rewritten_query + recent history; copyeditor constraints present")

    # ── Test 10: formatter polarity safety check ─────────────────
    _reset_sid()
    positive_draft = "Noa Levi's salary is in the $85,000 - $110,000 range based on the employees table."
    inverted       = "There is no employee named Noa Levi in the system."
    inv_resp = MagicMock(); inv_resp.content = inverted
    inv_llm  = MagicMock(); inv_llm.ainvoke = AsyncMock(return_value=inv_resp)
    inv_state = {
        **base_state,
        "final_answer":    positive_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
        "rewritten_query": "What is Noa Levi's salary?",
    }
    result_inv = _run(inv_state, inv_llm)
    assert result_inv["final_answer"] == positive_draft, \
        "Safety check must revert to audited draft on polarity inversion"
    print("PASS: formatter polarity inversion triggers safety check")

    # Legitimate negation (draft already negates) → formatted output accepted
    negative_draft     = "No employee named Noa Levie was found in the records."
    negative_formatted = "There is no employee named Noa Levie."
    neg_resp = MagicMock(); neg_resp.content = negative_formatted
    neg_llm  = MagicMock(); neg_llm.ainvoke = AsyncMock(return_value=neg_resp)
    neg_state = {
        **base_state,
        "final_answer":    negative_draft,
        "audit_result":    {"verdict": "PASS", "notes": ""},
        "rewritten_query": "What is Noa Levie's salary?",
    }
    result_neg = _run(neg_state, neg_llm)
    assert result_neg["final_answer"] == negative_formatted
    print("PASS: legitimate negation in audited draft delivers formatted output")

    # ── Test 11: delivery rejects when audit verdict != PASS ─────
    _reset_sid()
    no_llm = MagicMock(); no_llm.ainvoke = AsyncMock()
    bad_state = {
        **base_state,
        "final_answer":  "Some answer that should not be delivered.",
        "audit_result":  {"verdict": "FAIL", "notes": "Rejected."},
    }
    result_bad_audit = _run(bad_state, no_llm)
    no_llm.ainvoke.assert_not_called()
    assert result_bad_audit["final_answer"] == _FAILURE_MESSAGE
    print("PASS: delivery with non-PASS audit returns _FAILURE_MESSAGE, no LLM call")

    # ── Test 12: reasoning prompt receives data_context + session_context + history ──
    _reset_sid()
    seen_user_msg: list = []
    seen_resp = MagicMock()
    seen_resp.content = _reasoning_payload("PLAN", rewritten_query="What is X?")
    async def capture_seen(messages):
        seen_user_msg.extend(messages)
        return seen_resp
    seen_llm = MagicMock(); seen_llm.ainvoke = capture_seen

    # Pre-populate session context
    update_session_context("test-session-001", {"user_name": "Taylor"})

    seen_state = {
        **base_state,
        "original_query": "what is X?",
        "conversation_history": [{"role": "user", "content": "earlier message"}],
    }
    _run(seen_state, seen_llm)
    user_prompt = next(m["content"] for m in seen_user_msg if m["role"] == "user")
    system_prompt = next(m["content"] for m in seen_user_msg if m["role"] == "system")
    assert "DATA CONTEXT:"                in user_prompt, "reasoning prompt must include DATA CONTEXT block"
    assert fake_data_context.splitlines()[0] in user_prompt, "data_context content must reach the prompt"
    assert "SESSION CONTEXT"              in user_prompt, "reasoning prompt must include SESSION CONTEXT block"
    assert "Taylor"                        in user_prompt, "SESSION CONTEXT must carry prior user_name"
    assert "CONVERSATION HISTORY"         in user_prompt, "reasoning prompt must include history"
    assert "CURRENT USER MESSAGE"         in user_prompt
    # System prompt must not carry the removed classification vocabulary.
    for removed in ("conversation_state", "intent_type", '"fresh"', "post_clarification"):
        assert removed not in system_prompt, \
            f"reasoning system prompt still references removed token {removed!r}"
    print("PASS: reasoning prompt includes context blocks and drops removed classification tokens")

    # ── Test 13: history truncation — last MAX_REASONING_HISTORY entries only ──
    _reset_sid()
    big_history = []
    for i in range(20):
        big_history.append({"role": "user",      "content": f"user-msg-{i}"})
        big_history.append({"role": "assistant", "content": f"asst-msg-{i}"})  # 40 messages total

    cap_user: list = []
    trunc_resp = MagicMock()
    trunc_resp.content = _reasoning_payload("PLAN", rewritten_query="What?")
    async def cap_trunc(messages):
        cap_user.extend(messages)
        return trunc_resp
    trunc_llm = MagicMock(); trunc_llm.ainvoke = cap_trunc

    _run({**base_state, "conversation_history": big_history, "original_query": "What?"}, trunc_llm)
    user_prompt = next(m["content"] for m in cap_user if m["role"] == "user")
    # MAX_REASONING_HISTORY = 12 → last 12 messages. We had 40 → last 12 are
    # user-msg-14, asst-msg-14, user-msg-15, ..., asst-msg-19. So user-msg-0
    # must NOT appear, but asst-msg-19 MUST.
    assert "user-msg-0"   not in user_prompt
    assert "user-msg-13"  not in user_prompt
    assert "user-msg-14"  in user_prompt
    assert "asst-msg-19"  in user_prompt
    print(f"PASS: reasoning call truncates history to last {MAX_REASONING_HISTORY} messages")

    # ── Test 14: post-clarification "yes" → PLAN rewrites full intent ──
    _reset_sid()
    yes_json = _reasoning_payload(
        "PLAN",
        rewritten_query="What is the salary range for Noa Levy?",
    )
    yes_state = {
        **base_state,
        "original_query": "yes",
        "conversation_history": [
            {"role": "user",      "content": "What is the salary range for Noa Levi?"},
            {"role": "assistant", "content": "I could not find an employee named Noa Levi. Did you mean a similar name?"},
            {"role": "user",      "content": "maybe noa levy"},
            {"role": "assistant", "content": "Did you mean Noa Levy (not Levi)?"},
        ],
    }
    result_yes = _run(yes_state, _mock_llm(yes_json))
    assert result_yes["chat_intent"] == "PLAN"
    assert "Noa Levy" in result_yes["rewritten_query"]
    assert "salary"   in result_yes["rewritten_query"].lower()
    print("PASS: post-clarification 'yes' routes as PLAN with reconstructed intent")

    # ── Test 15: identity + question in one message → PLAN + context update ──
    _reset_sid()
    combo_json = _reasoning_payload(
        "PLAN",
        rewritten_query="What is Dan's (Engineering) salary?",
        session_context_update={"user_name": "Dan", "department": "Engineering"},
    )
    combo_state = {**base_state, "original_query": "I'm Dan from Engineering, what's my salary?"}
    result_combo = _run(combo_state, _mock_llm(combo_json))
    assert result_combo["chat_intent"]     == "PLAN"
    assert result_combo["rewritten_query"] == "What is Dan's (Engineering) salary?"
    ctx = get_session_context("test-session-001")
    assert ctx["user_name"]  == "Dan"
    assert ctx["department"] == "Engineering"
    print("PASS: identity + question emits PLAN + session_context_update together")

    # ── Test 16: delivery preserves chat_reasoning set on prior turn ──
    # Delivery path returns only {conversation_history, final_answer}, not
    # chat_reasoning. chat_reasoning is a snapshot of the entry-turn decision
    # and must not be overwritten by the delivery path on a retry loop.
    _reset_sid()
    plain_draft     = "Dan Cohen earns $95,000 annually based on the employees table."
    plain_formatted = "Dan Cohen makes $95,000 per year."
    plain_resp = MagicMock(); plain_resp.content = plain_formatted
    plain_llm  = MagicMock(); plain_llm.ainvoke = AsyncMock(return_value=plain_resp)
    plain_state = {
        **base_state,
        "chat_reasoning":  '{"previous": "reasoning"}',
        "final_answer":    plain_draft,
        "audit_result":    {"verdict": "PASS", "notes": "Verified."},
        "rewritten_query": "What is Dan Cohen's salary?",
    }
    result_plain = _run(plain_state, plain_llm)
    assert "chat_reasoning" not in result_plain, \
        "delivery path must not overwrite chat_reasoning"
    print("PASS: delivery path returns only {conversation_history, final_answer}")

    # ── Test 17: _normalize_reasoning / _fallback_reasoning emit the trimmed schema ──
    _reset_sid()
    normalized = _normalize_reasoning({
        "reasoning": {
            "user_intent":        "look up a fact",
            "gaps":               [],
            "decision_rationale": "clear lookup",
        },
        "decision":               "PLAN",
        "rewritten_query":        "What is Dan Cohen's salary?",
        "clarifying_question":    "",
        "direct_response":        "",
        "session_context_update": {},
    }, "original")
    assert set(normalized["reasoning"].keys()) == {"user_intent", "gaps", "decision_rationale"}, \
        f"normalized reasoning has unexpected keys: {normalized['reasoning'].keys()}"

    fb = _fallback_reasoning("orig q")
    assert set(fb["reasoning"].keys()) == {"user_intent", "gaps", "decision_rationale"}, \
        f"fallback reasoning has unexpected keys: {fb['reasoning'].keys()}"
    assert fb["decision"] == "PLAN"
    assert fb["rewritten_query"] == "orig q"
    print("PASS: reasoning normalize/fallback emit only {user_intent, gaps, decision_rationale}")

    # ── Test 18: PLAN rewritten_query is natural language, no procedural words ──
    # Test fixtures we author in this suite must obey the contract the prompt
    # enforces at runtime. If this ever fails, rewrite the test fixture to
    # match how a user would actually phrase the question.
    _reset_sid()
    banned_tokens = (
        "source", "table", "record", "document", "file",
        "retrieve", "provide", "lookup", "if available", "otherwise",
    )
    clean_queries_seen: list[str] = []
    # Collect every rewritten_query authored in _reasoning_payload calls above.
    for q in (
        "What is Dan Cohen's flight class entitlement?",
        "What is the salary range for Noa Levy?",
        "What is Dan's (Engineering) salary?",
        "What is Dan Cohen's salary?",
        "What is X?",
    ):
        clean_queries_seen.append(q)

    for q in clean_queries_seen:
        low = q.lower()
        for banned in banned_tokens:
            assert banned not in low, (
                f"test-authored rewritten_query {q!r} contains banned token "
                f"{banned!r} — update the fixture to a natural-language question"
            )
    print("PASS: all test-authored PLAN rewrites avoid source/procedural language")

    _reset_sid()
    print("\nPASS: all chat tests passed")


if __name__ == "__main__":
    test_chat()
